"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class biped_trunk_state_t(object):
    __slots__ = ["timestamp", "finished", "base_p", "base_pd", "base_pdd", "base_rpy", "base_rpyd", "base_rpydd", "l_p", "r_p", "l_pd", "r_pd", "l_pdd", "r_pdd", "l_contact", "r_contact", "l_f", "r_f"]

    def __init__(self):
        self.timestamp = 0.0
        self.finished = False
        self.base_p = [ 0.0 for dim0 in range(3) ]
        self.base_pd = [ 0.0 for dim0 in range(3) ]
        self.base_pdd = [ 0.0 for dim0 in range(3) ]
        self.base_rpy = [ 0.0 for dim0 in range(3) ]
        self.base_rpyd = [ 0.0 for dim0 in range(3) ]
        self.base_rpydd = [ 0.0 for dim0 in range(3) ]
        self.l_p = [ 0.0 for dim0 in range(3) ]
        self.r_p = [ 0.0 for dim0 in range(3) ]
        self.l_pd = [ 0.0 for dim0 in range(3) ]
        self.r_pd = [ 0.0 for dim0 in range(3) ]
        self.l_pdd = [ 0.0 for dim0 in range(3) ]
        self.r_pdd = [ 0.0 for dim0 in range(3) ]
        self.l_contact = False
        self.r_contact = False
        self.l_f = [ 0.0 for dim0 in range(3) ]
        self.r_f = [ 0.0 for dim0 in range(3) ]

    def encode(self):
        buf = BytesIO()
        buf.write(biped_trunk_state_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">db", self.timestamp, self.finished))
        buf.write(struct.pack('>3d', *self.base_p[:3]))
        buf.write(struct.pack('>3d', *self.base_pd[:3]))
        buf.write(struct.pack('>3d', *self.base_pdd[:3]))
        buf.write(struct.pack('>3d', *self.base_rpy[:3]))
        buf.write(struct.pack('>3d', *self.base_rpyd[:3]))
        buf.write(struct.pack('>3d', *self.base_rpydd[:3]))
        buf.write(struct.pack('>3d', *self.l_p[:3]))
        buf.write(struct.pack('>3d', *self.r_p[:3]))
        buf.write(struct.pack('>3d', *self.l_pd[:3]))
        buf.write(struct.pack('>3d', *self.r_pd[:3]))
        buf.write(struct.pack('>3d', *self.l_pdd[:3]))
        buf.write(struct.pack('>3d', *self.r_pdd[:3]))
        buf.write(struct.pack(">bb", self.l_contact, self.r_contact))
        buf.write(struct.pack('>3d', *self.l_f[:3]))
        buf.write(struct.pack('>3d', *self.r_f[:3]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != biped_trunk_state_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return biped_trunk_state_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = biped_trunk_state_t()
        self.timestamp = struct.unpack(">d", buf.read(8))[0]
        self.finished = bool(struct.unpack('b', buf.read(1))[0])
        self.base_p = struct.unpack('>3d', buf.read(24))
        self.base_pd = struct.unpack('>3d', buf.read(24))
        self.base_pdd = struct.unpack('>3d', buf.read(24))
        self.base_rpy = struct.unpack('>3d', buf.read(24))
        self.base_rpyd = struct.unpack('>3d', buf.read(24))
        self.base_rpydd = struct.unpack('>3d', buf.read(24))
        self.l_p = struct.unpack('>3d', buf.read(24))
        self.r_p = struct.unpack('>3d', buf.read(24))
        self.l_pd = struct.unpack('>3d', buf.read(24))
        self.r_pd = struct.unpack('>3d', buf.read(24))
        self.l_pdd = struct.unpack('>3d', buf.read(24))
        self.r_pdd = struct.unpack('>3d', buf.read(24))
        self.l_contact = bool(struct.unpack('b', buf.read(1))[0])
        self.r_contact = bool(struct.unpack('b', buf.read(1))[0])
        self.l_f = struct.unpack('>3d', buf.read(24))
        self.r_f = struct.unpack('>3d', buf.read(24))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if biped_trunk_state_t in parents: return 0
        tmphash = (0x9e147618af8cf1b2) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if biped_trunk_state_t._packed_fingerprint is None:
            biped_trunk_state_t._packed_fingerprint = struct.pack(">Q", biped_trunk_state_t._get_hash_recursive([]))
        return biped_trunk_state_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

