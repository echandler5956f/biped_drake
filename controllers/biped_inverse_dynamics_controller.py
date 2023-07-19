from controllers.biped_basic_controller import *

class BipedIDController(BipedBasicController):
    """
    A standard QP-based inverse dynamics controller. 

    Takes as input desired positions/velocities/accelerations of the 
    feet and floating base and computes corresponding joint torques. 
    """
    def __init__(self, plant, dt, use_lcm=False):
        BipedBasicController.__init__(self, plant, dt, use_lcm=use_lcm)

        # inputs from the trunk model are sent in a dictionary
        self.DeclareAbstractInputPort(
                "trunk_input",
                AbstractValue.Make({}))  
        
        # Set the friction coefficient
        self.mu = 0.7

        # Choose a solver
        #self.solver = GurobiSolver()
        self.solver = OsqpSolver()
    
    def AddJacobianTypeCost(self, J, qdd, Jd_qd, xdd_des, weight=1.0):
        """
        Add a quadratic cost of the form
            weight*| J*qdd + Jd_qd - xdd_des |^2
        to the whole-body controller QP.
        """
        # Put in the form 1/2*qdd'*Q*qdd + c'*qdd for fast formulation
        Q = weight*np.dot(J.T,J)
        c = weight*(np.dot(Jd_qd.T,J) - np.dot(xdd_des.T,J)).T

        return self.mp.AddQuadraticCost(Q,c,qdd, is_convex=True)

    def AddJacobianTypeConstraint(self, J, qdd, Jd_qd, xdd_des):
        """
        Add a linear constraint of the form
            J*qdd + Jd_qd == xdd_des
        to the whole-body controller QP.
        """
        A_eq = J     # A_eq*qdd == b_eq
        b_eq = xdd_des-Jd_qd

        return self.mp.AddLinearEqualityConstraint(A_eq, b_eq, qdd)

    def AddDynamicsConstraint(self, M, qdd, C, tau_g, S, tau, J_c, f_c):
        """
        Add a dynamics constraint of the form
            M*qdd + Cv + tau_g == S'*tau + sum(J_c[i]'*f_c[i])
        to the whole-body controller QP.
        """
        # We'll rewrite the constraints in the form A_eq*x == b_eq for speed
        A_eq = np.hstack([M, -S.T])
        x = np.vstack([qdd,tau])

        for j in range(len(J_c)):
            A_eq = np.hstack([A_eq, -J_c[j].T])
            x = np.vstack([x,f_c[j]])

        b_eq = -C-tau_g

        return self.mp.AddLinearEqualityConstraint(A_eq,b_eq,x)

    def AddFrictionPyramidConstraint(self, f_c):
        """
        Add a friction pyramid constraint for the given set of contact forces
        to the whole-body controller QP.
        """
        num_contacts = len(f_c)

        A_i = np.asarray([[ 1, 0, -self.mu],   # pyramid approximation of CWC for one
                          [-1, 0, -self.mu]   # contact force f \in R^3
                          ])

        # We'll formulate as lb <= Ax <= ub, where x=[f_1',f_2',...]'
        A = np.kron(np.eye(num_contacts),A_i)

        ub = np.zeros((2*num_contacts,1))
        lb = -np.inf*np.ones((2*num_contacts,1))

        x = np.vstack([f_c[j] for j in range(num_contacts)])

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=x)

    def AddContactConstraint(self, J_c, vd, Jdv_c, v):
        """
        Add contact constraints with velocity damping

            J_c[j]*vd + Jdv_c[j] == -Kd*J_c[j]*v
        """
        Kd = 100*np.eye(3)

        num_contacts = len(J_c)
        for j in range(num_contacts):
            pd = (J_c[j]@v).reshape(3,1)
            pdd_des = -Kd@pd

            constraint = self.AddJacobianTypeConstraint(J_c[j], vd, Jdv_c[j], pdd_des)

    def ControlLaw(self, context, q, v):
        """
        A standard inverse dynamics whole-body QP:

           minimize:
               w_body* || J_body*vd + Jd_body*v - vd_body_des ||^2 +
               w_foot* || J_s*vd+ Jd_s*v - pdd_s_des ||^2
           subject to:
                M*vd + Cv + tau_g = S'*tau + sum(J'*f)
                f \in friction cones
                J_cj*vd + Jd_cj*v == 0
        """

        ######### Tuning Parameters #########
        Kp_body_p = 500.0
        Kd_body_p = 50.0

        Kp_body_rpy = Kp_body_p
        Kd_body_rpy = Kd_body_p

        Kp_foot = 100.0
        Kd_foot = 20.0

        w_body = 10.0
        w_foot = 1.0
        #####################################

        # Compute Dynamics Quantities
        M, Cv, tau_g, S = self.CalcDynamics()

        # Get setpoint data from the trunk model
        trunk_data = self.EvalAbstractInput(context,1).get_value()
        
        contact_feet = trunk_data["contact_states"]       # Note: it may be better to determine
        swing_feet = [not foot for foot in contact_feet]  # contact states from the actual robot rather than
        num_contact = sum(contact_feet)                   # the planned trunk trajectory.
        num_swing = sum(swing_feet)

        p_body_nom = trunk_data["p_body"]
        pd_body_nom = trunk_data["pd_body"]
        pdd_body_nom = trunk_data["pdd_body"]

        rpy_body_nom = trunk_data["rpy_body"]
        rpyd_body_nom = trunk_data["rpyd_body"]
        rpydd_body_nom = trunk_data["rpydd_body"]
        
        p_feet_nom = np.array([trunk_data["p_l"],trunk_data["p_r"]])
        pd_feet_nom = np.array([trunk_data["pd_l"],trunk_data["pd_r"]])
        pdd_feet_nom = np.array([trunk_data["pdd_l"],trunk_data["pdd_r"]])

        p_s_nom = p_feet_nom[swing_feet]
        pd_s_nom = pd_feet_nom[swing_feet]
        pdd_s_nom = pdd_feet_nom[swing_feet]

        # Get robot's actual task-space (body pose + foot positions) data
        X_body, J_body, Jdv_body = self.CalcFramePoseQuantities(self.body_frame)

        p_body = X_body.translation()
        pd_body = (J_body@v)[3:]

        RPY_body = RollPitchYaw(X_body.rotation())  # RPY object helps convert between angular velocity and rpyd
        rpy_body = RPY_body.vector()
        omega_body = (J_body@v)[:3]   # angular velocity of the body
        rpyd_body = RPY_body.CalcRpyDtFromAngularVelocityInParent(omega_body)

        p_l, J_l, Jdv_l = self.CalcFramePositionQuantities(self.l_foot_frame)
        p_r, J_r, Jdv_r = self.CalcFramePositionQuantities(self.r_foot_frame)

        p_feet = np.array([p_l, p_r]).reshape(2,3)
        J_feet = np.array([J_l, J_r])
        Jdv_feet = np.array([Jdv_l, Jdv_r])
        pd_feet = J_feet@v

        p_s = p_feet[swing_feet]
        pd_s = pd_feet[swing_feet]

        J_c = J_feet[contact_feet]
        J_s = J_feet[swing_feet]
        Jdv_c = Jdv_feet[contact_feet]
        Jdv_s = Jdv_feet[swing_feet]

        # Set desired task-space accelerations
        pdd_body_des = pdd_body_nom - Kp_body_p*(p_body - p_body_nom) \
                                    - Kd_body_p*(pd_body - pd_body_nom)

        rpydd_body_des = rpydd_body_nom - Kp_body_rpy*(rpy_body - rpy_body_nom) \
                                        - Kd_body_rpy*(rpyd_body - rpyd_body_nom)
        omegad_body_des = RPY_body.CalcAngularVelocityInParentFromRpyDt(rpydd_body_des)
        
        vd_body_des = np.hstack([omegad_body_des,pdd_body_des])   # desired spatial acceleration of the body

        pdd_s_des = pdd_s_nom  - Kp_foot*(p_s - p_s_nom) \
                               - Kd_foot*(pd_s - pd_s_nom)

        # Set up the QP
        self.mp = MathematicalProgram()
        
        vd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'vd')
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        f_c = [self.mp.NewContinuousVariables(3,1,'f_%s'%j) for j in range(num_contact)]

        # min || J_body*vd + Jd_body*v - pdd_body_des ||^2
        self.AddJacobianTypeCost(J_body, vd, Jdv_body, vd_body_des, weight=w_body)

        # min || J_s*vd+ Jd_s*v - pdd_s_des ||^2
        for i in range(num_swing):
            self.AddJacobianTypeCost(J_s[i], vd, Jdv_s[i], pdd_s_des[i], weight=w_foot)

        # s.t.  M*vd + Cv + tau_g = S'*tau + sum(J_c[j]'*f_c[j])
        self.AddDynamicsConstraint(M, vd, Cv, tau_g, S, tau, J_c, f_c)

        if any(contact_feet):
            # s.t. f_c[j] in friction cones
            self.AddFrictionPyramidConstraint(f_c)

            # s.t. J_cj*vd + Jd_cj*v == 0 (+ some daming)
            self.AddContactConstraint(J_c, vd, Jdv_c, v)

        result = self.solver.Solve(self.mp)
        print(result.GetSolution(tau))
        assert result.is_success()
        tau = result.GetSolution(tau)

        # Set error for logging
        x_tilde = np.hstack([rpy_body - rpy_body_nom,
                             p_body - p_body_nom,
                             p_s.flatten() - p_s_nom.flatten()])
        self.err = x_tilde.T@x_tilde
        self.res = result.get_solver_details().primal_res
    
        return tau
