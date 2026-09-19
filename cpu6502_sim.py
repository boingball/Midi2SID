"""A small, documented-opcode-only 6502 interpreter for offline testing.

This exists because the Midi2SID sandbox has no VICE (or any C64 emulator)
installed, so there is no way to visually verify hand-assembled 6502/VIC-II
code before it ships. This lets us at least execute the generated machine
code against a plain RAM array and inspect the resulting memory (screen,
bitmap, SID registers) programmatically or by rendering it to a PNG.

Only the addressing modes and mnemonics actually needed by build_prg.py are
implemented. Unknown opcodes raise, which is exactly what we want during
development (it means an instruction was hand-encoded but never exercised).
"""
from __future__ import annotations


class CPU:
    def __init__(self, mem: bytearray | None = None, char_rom: bytes | None = None):
        self.mem = mem if mem is not None else bytearray(65536)
        self.a = self.x = self.y = 0
        self.sp = 0xff
        self.pc = 0
        self.c = self.z = self.n = self.v = 0
        self.steps = 0
        self.io_hook = None  # optional callable(addr, value) for writes
        # $D000-$DFFF is either I/O or character ROM depending on the CPU
        # port ($01) CHAREN bit, matching real hardware: a write there
        # always lands in the shadow RAM (self.mem, exactly like any other
        # write), but a *read* while CHAREN is low must come from character
        # ROM, not whatever shadow RAM holds - conflating the two let a
        # write meant for the VIC/SID registers silently "corrupt" character
        # ROM in this simulator, something that cannot happen on real
        # hardware (ROM is physically read-only).
        self.char_rom = char_rom
        self.charen = True  # I/O visible; matches the KERNAL's boot default

    # -- memory helpers ---------------------------------------------------
    def rd(self, addr: int) -> int:
        addr &= 0xffff
        if not self.charen and self.char_rom is not None and 0xd000 <= addr <= 0xdfff:
            return self.char_rom[addr - 0xd000]
        return self.mem[addr]

    def wr(self, addr: int, value: int) -> None:
        addr &= 0xffff
        value &= 0xff
        self.mem[addr] = value
        if addr == 0x01:
            self.charen = bool(value & 0x04)
        if self.io_hook is not None:
            self.io_hook(addr, value)

    def rd16(self, addr: int) -> int:
        return self.rd(addr) | (self.rd(addr + 1) << 8)

    def fetch(self) -> int:
        value = self.rd(self.pc)
        self.pc = (self.pc + 1) & 0xffff
        return value

    def fetch16(self) -> int:
        lo = self.fetch()
        hi = self.fetch()
        return lo | (hi << 8)

    # -- flag helpers -------------------------------------------------------
    def set_zn(self, value: int) -> int:
        value &= 0xff
        self.z = 1 if value == 0 else 0
        self.n = 1 if value & 0x80 else 0
        return value

    # -- addressing modes: return an address (or None for accumulator) ------
    def am_imm(self):
        addr = self.pc
        self.pc = (self.pc + 1) & 0xffff
        return addr

    def am_zp(self):
        return self.fetch()

    def am_zpx(self):
        return (self.fetch() + self.x) & 0xff

    def am_zpy(self):
        return (self.fetch() + self.y) & 0xff

    def am_abs(self):
        return self.fetch16()

    def am_absx(self):
        return (self.fetch16() + self.x) & 0xffff

    def am_absy(self):
        return (self.fetch16() + self.y) & 0xffff

    def am_indx(self):
        base = (self.fetch() + self.x) & 0xff
        return self.rd(base) | (self.rd((base + 1) & 0xff) << 8)

    def am_indy(self):
        base = self.fetch()
        addr = self.rd(base) | (self.rd((base + 1) & 0xff) << 8)
        return (addr + self.y) & 0xffff

    def am_ind(self):
        # 6502 JMP (indirect) page-boundary bug intentionally NOT modelled
        # since our code never places such a vector on a page boundary.
        ptr = self.fetch16()
        return self.rd(ptr) | (self.rd(ptr + 1) << 8)

    # -- run ------------------------------------------------------------
    def run(self, start: int, max_steps: int = 2_000_000, stop_at: set[int] | None = None) -> None:
        self.pc = start
        stop_at = stop_at or set()
        executed = 0
        while executed < max_steps:
            if self.pc in stop_at:
                return
            self.step()
            executed += 1
            if executed >= max_steps:
                raise RuntimeError(f"6502 sim exceeded {max_steps} steps (runaway loop?) at pc=${self.pc:04x}")

    def step(self) -> None:
        self.steps += 1
        op = self.fetch()
        handler = OPCODES.get(op)
        if handler is None:
            raise ValueError(f"unimplemented opcode ${op:02x} at ${self.pc - 1:04x}")
        handler(self)

    # -- branch helper ----------------------------------------------------
    def branch(self, condition: bool) -> None:
        offset = self.fetch()
        if offset & 0x80:
            offset -= 0x100
        if condition:
            self.pc = (self.pc + offset) & 0xffff

    def push(self, value: int) -> None:
        self.mem[0x100 + self.sp] = value & 0xff
        self.sp = (self.sp - 1) & 0xff

    def pop(self) -> int:
        self.sp = (self.sp + 1) & 0xff
        return self.mem[0x100 + self.sp]


def _adc(cpu: CPU, value: int) -> None:
    total = cpu.a + value + cpu.c
    cpu.v = 1 if (~(cpu.a ^ value) & (cpu.a ^ total) & 0x80) else 0
    cpu.c = 1 if total > 0xff else 0
    cpu.a = cpu.set_zn(total)


def _sbc(cpu: CPU, value: int) -> None:
    _adc(cpu, (~value) & 0xff)


OPCODES = {}


def op(*codes):
    def register(fn):
        for code in codes:
            OPCODES[code] = fn
        return fn
    return register


# Loads / stores
@op(0xa9)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_imm()))


@op(0xa5)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_zp()))


@op(0xb5)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_zpx()))


@op(0xad)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_abs()))


@op(0xbd)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_absx()))


@op(0xb9)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_absy()))


@op(0xa1)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_indx()))


@op(0xb1)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.rd(cpu.am_indy()))


@op(0xa2)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.rd(cpu.am_imm()))


@op(0xa6)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.rd(cpu.am_zp()))


@op(0xb6)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.rd(cpu.am_zpy()))


@op(0xae)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.rd(cpu.am_abs()))


@op(0xbe)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.rd(cpu.am_absy()))


@op(0xa0)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.rd(cpu.am_imm()))


@op(0xa4)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.rd(cpu.am_zp()))


@op(0xb4)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.rd(cpu.am_zpx()))


@op(0xac)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.rd(cpu.am_abs()))


@op(0xbc)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.rd(cpu.am_absx()))


@op(0x85)
def _(cpu):
    cpu.wr(cpu.am_zp(), cpu.a)


@op(0x95)
def _(cpu):
    cpu.wr(cpu.am_zpx(), cpu.a)


@op(0x8d)
def _(cpu):
    cpu.wr(cpu.am_abs(), cpu.a)


@op(0x9d)
def _(cpu):
    cpu.wr(cpu.am_absx(), cpu.a)


@op(0x99)
def _(cpu):
    cpu.wr(cpu.am_absy(), cpu.a)


@op(0x81)
def _(cpu):
    cpu.wr(cpu.am_indx(), cpu.a)


@op(0x91)
def _(cpu):
    cpu.wr(cpu.am_indy(), cpu.a)


@op(0x86)
def _(cpu):
    cpu.wr(cpu.am_zp(), cpu.x)


@op(0x96)
def _(cpu):
    cpu.wr(cpu.am_zpy(), cpu.x)


@op(0x8e)
def _(cpu):
    cpu.wr(cpu.am_abs(), cpu.x)


@op(0x84)
def _(cpu):
    cpu.wr(cpu.am_zp(), cpu.y)


@op(0x94)
def _(cpu):
    cpu.wr(cpu.am_zpx(), cpu.y)


@op(0x8c)
def _(cpu):
    cpu.wr(cpu.am_abs(), cpu.y)


# Transfers
@op(0xaa)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.a)


@op(0xa8)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.a)


@op(0x8a)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.x)


@op(0x98)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.y)


@op(0x9a)
def _(cpu):
    cpu.sp = cpu.x


@op(0xba)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.sp)


# Stack
@op(0x48)
def _(cpu):
    cpu.push(cpu.a)


@op(0x68)
def _(cpu):
    cpu.a = cpu.set_zn(cpu.pop())


# Increment / decrement
@op(0xe8)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.x + 1)


@op(0xca)
def _(cpu):
    cpu.x = cpu.set_zn(cpu.x - 1)


@op(0xc8)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.y + 1)


@op(0x88)
def _(cpu):
    cpu.y = cpu.set_zn(cpu.y - 1)


@op(0xe6)
def _(cpu):
    addr = cpu.am_zp()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) + 1))


@op(0xf6)
def _(cpu):
    addr = cpu.am_zpx()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) + 1))


@op(0xee)
def _(cpu):
    addr = cpu.am_abs()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) + 1))


@op(0xfe)
def _(cpu):
    addr = cpu.am_absx()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) + 1))


@op(0xc6)
def _(cpu):
    addr = cpu.am_zp()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) - 1))


@op(0xd6)
def _(cpu):
    addr = cpu.am_zpx()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) - 1))


@op(0xce)
def _(cpu):
    addr = cpu.am_abs()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) - 1))


@op(0xde)
def _(cpu):
    addr = cpu.am_absx()
    cpu.wr(addr, cpu.set_zn(cpu.rd(addr) - 1))


# Logic / arithmetic (accumulator + immediate/zp/abs variants)
def _make_alu(name, modes, fn):
    for opcode, mode in modes.items():
        def handler(cpu, mode=mode, fn=fn):
            addr = mode(cpu)
            fn(cpu, cpu.rd(addr))
        OPCODES[opcode] = handler


_make_alu("AND", {0x29: CPU.am_imm, 0x25: CPU.am_zp, 0x35: CPU.am_zpx, 0x2d: CPU.am_abs,
                   0x3d: CPU.am_absx, 0x39: CPU.am_absy, 0x21: CPU.am_indx, 0x31: CPU.am_indy},
          lambda cpu, v: setattr(cpu, "a", cpu.set_zn(cpu.a & v)))

_make_alu("ORA", {0x09: CPU.am_imm, 0x05: CPU.am_zp, 0x15: CPU.am_zpx, 0x0d: CPU.am_abs,
                   0x1d: CPU.am_absx, 0x19: CPU.am_absy, 0x01: CPU.am_indx, 0x11: CPU.am_indy},
          lambda cpu, v: setattr(cpu, "a", cpu.set_zn(cpu.a | v)))

_make_alu("EOR", {0x49: CPU.am_imm, 0x45: CPU.am_zp, 0x55: CPU.am_zpx, 0x4d: CPU.am_abs,
                   0x5d: CPU.am_absx, 0x59: CPU.am_absy, 0x41: CPU.am_indx, 0x51: CPU.am_indy},
          lambda cpu, v: setattr(cpu, "a", cpu.set_zn(cpu.a ^ v)))

_make_alu("ADC", {0x69: CPU.am_imm, 0x65: CPU.am_zp, 0x75: CPU.am_zpx, 0x6d: CPU.am_abs,
                   0x7d: CPU.am_absx, 0x79: CPU.am_absy, 0x61: CPU.am_indx, 0x71: CPU.am_indy},
          _adc)

_make_alu("SBC", {0xe9: CPU.am_imm, 0xe5: CPU.am_zp, 0xf5: CPU.am_zpx, 0xed: CPU.am_abs,
                   0xfd: CPU.am_absx, 0xf9: CPU.am_absy, 0xe1: CPU.am_indx, 0xf1: CPU.am_indy},
          _sbc)


def _cmp(cpu, reg_name, value):
    reg = getattr(cpu, reg_name)
    result = (reg - value) & 0x1ff
    cpu.c = 1 if reg >= value else 0
    cpu.set_zn(result)


_make_alu("CMP", {0xc9: CPU.am_imm, 0xc5: CPU.am_zp, 0xd5: CPU.am_zpx, 0xcd: CPU.am_abs,
                   0xdd: CPU.am_absx, 0xd9: CPU.am_absy, 0xc1: CPU.am_indx, 0xd1: CPU.am_indy},
          lambda cpu, v: _cmp(cpu, "a", v))
_make_alu("CPX", {0xe0: CPU.am_imm, 0xe4: CPU.am_zp, 0xec: CPU.am_abs}, lambda cpu, v: _cmp(cpu, "x", v))
_make_alu("CPY", {0xc0: CPU.am_imm, 0xc4: CPU.am_zp, 0xcc: CPU.am_abs}, lambda cpu, v: _cmp(cpu, "y", v))


@op(0x24)
def _(cpu):
    v = cpu.rd(cpu.am_zp())
    cpu.z = 1 if (cpu.a & v) == 0 else 0
    cpu.n = 1 if v & 0x80 else 0
    cpu.v = 1 if v & 0x40 else 0


@op(0x2c)
def _(cpu):
    v = cpu.rd(cpu.am_abs())
    cpu.z = 1 if (cpu.a & v) == 0 else 0
    cpu.n = 1 if v & 0x80 else 0
    cpu.v = 1 if v & 0x40 else 0


# Shifts / rotates (accumulator and memory forms)
def _shift(cpu, addr, fn):
    if addr is None:
        cpu.a = fn(cpu, cpu.a)
    else:
        cpu.wr(addr, fn(cpu, cpu.rd(addr)))


def _asl(cpu, v):
    cpu.c = 1 if v & 0x80 else 0
    return cpu.set_zn(v << 1)


def _lsr(cpu, v):
    cpu.c = v & 1
    return cpu.set_zn(v >> 1)


def _rol(cpu, v):
    carry_in = cpu.c
    cpu.c = 1 if v & 0x80 else 0
    return cpu.set_zn(((v << 1) | carry_in) & 0xff)


def _ror(cpu, v):
    carry_in = cpu.c
    cpu.c = v & 1
    return cpu.set_zn((v >> 1) | (carry_in << 7))


@op(0x0a)
def _(cpu):
    _shift(cpu, None, _asl)


@op(0x06)
def _(cpu):
    _shift(cpu, cpu.am_zp(), _asl)


@op(0x0e)
def _(cpu):
    _shift(cpu, cpu.am_abs(), _asl)


@op(0x16)
def _(cpu):
    _shift(cpu, cpu.am_zpx(), _asl)


@op(0x4a)
def _(cpu):
    _shift(cpu, None, _lsr)


@op(0x46)
def _(cpu):
    _shift(cpu, cpu.am_zp(), _lsr)


@op(0x4e)
def _(cpu):
    _shift(cpu, cpu.am_abs(), _lsr)


@op(0x2a)
def _(cpu):
    _shift(cpu, None, _rol)


@op(0x26)
def _(cpu):
    _shift(cpu, cpu.am_zp(), _rol)


@op(0x2e)
def _(cpu):
    _shift(cpu, cpu.am_abs(), _rol)


@op(0x6a)
def _(cpu):
    _shift(cpu, None, _ror)


@op(0x66)
def _(cpu):
    _shift(cpu, cpu.am_zp(), _ror)


@op(0x6e)
def _(cpu):
    _shift(cpu, cpu.am_abs(), _ror)


# Flags
@op(0x18)
def _(cpu):
    cpu.c = 0


@op(0x38)
def _(cpu):
    cpu.c = 1


@op(0xd8)
def _(cpu):
    pass  # CLD: no decimal mode implemented, nothing to clear


@op(0xf8)
def _(cpu):
    pass  # SED: unused by this codebase


@op(0x58)
def _(cpu):
    pass  # CLI: no interrupts modelled


@op(0x78)
def _(cpu):
    pass  # SEI


@op(0xb8)
def _(cpu):
    cpu.v = 0


# Branches
@op(0xf0)
def _(cpu):
    cpu.branch(cpu.z == 1)


@op(0xd0)
def _(cpu):
    cpu.branch(cpu.z == 0)


@op(0x10)
def _(cpu):
    cpu.branch(cpu.n == 0)


@op(0x30)
def _(cpu):
    cpu.branch(cpu.n == 1)


@op(0x90)
def _(cpu):
    cpu.branch(cpu.c == 0)


@op(0xb0)
def _(cpu):
    cpu.branch(cpu.c == 1)


@op(0x50)
def _(cpu):
    cpu.branch(cpu.v == 0)


@op(0x70)
def _(cpu):
    cpu.branch(cpu.v == 1)


# Jumps / calls
@op(0x4c)
def _(cpu):
    cpu.pc = cpu.am_abs()


@op(0x6c)
def _(cpu):
    cpu.pc = cpu.am_ind()


@op(0x20)
def _(cpu):
    target = cpu.am_abs()
    ret = (cpu.pc - 1) & 0xffff
    cpu.push(ret >> 8)
    cpu.push(ret & 0xff)
    cpu.pc = target


@op(0x60)
def _(cpu):
    lo = cpu.pop()
    hi = cpu.pop()
    cpu.pc = (((hi << 8) | lo) + 1) & 0xffff


@op(0xea)
def _(cpu):
    pass  # NOP
