# gfx_v2 on-device front-end kernel — build to .vxbin.
#
# Compiles the setup -> binning front end (setup_k / binning_k) into a Vortex
# KMU kernel image vortexpipe loads at runtime. The toolchain invocation
# mirrors vp_compile.c (the driver's NIR-shader path) and tests/regression/
# common.mk: llvm-vortex clang with +xvortex/+zicond, the XLEN-matched GNU
# sysroot, link32.ld, libvortex2.a, then vxbin.py packages the ELF.
#
# Paths come from the same env the driver uses:
#   VORTEX_HOME    Vortex source tree   (sw/kernel/{scripts,include})
#   VORTEX_BUILD   Vortex build tree    (sw/kernel/libvortex2.a); default $(VORTEX_HOME)/build
#   VORTEX_TOOLDIR llvm-vortex + GNU toolchain + libc/libcrt
# XLEN is 32 (the Vortex graphics target).

XLEN          ?= 32
VORTEX_HOME   ?= $(HOME)/dev/vortex_v3/prism_v3
VORTEX_BUILD  ?= $(VORTEX_HOME)/build
VORTEX_TOOLDIR?= $(HOME)/vortex_rtu_tools

# Artifact directory. Defaults to in-tree (standalone `make`); the meson
# custom_target passes OUTDIR=<build dir> for an out-of-tree build.
OUTDIR        ?= .

TARGET   := riscv$(XLEN)-unknown-elf
GNU_DIR  := riscv$(XLEN)-gnu-toolchain
MARCH    := rv32imaf
MABI     := ilp32f
LINKER   := link$(XLEN).ld
LIBC     := libc$(XLEN)
LIBCRT   := libcrt$(XLEN)
CRT_A    := libclang_rt.builtins-riscv$(XLEN).a

CLANG    := $(VORTEX_TOOLDIR)/llvm-vortex/bin/clang
OBJCOPY  := $(VORTEX_TOOLDIR)/llvm-vortex/bin/llvm-objcopy

# -ffp-contract=off keeps the device setup math bit-faithful to the host
# Binning() oracle the front end was validated against.
VX_CFLAGS := --target=$(TARGET) \
  --sysroot=$(VORTEX_TOOLDIR)/$(GNU_DIR)/$(TARGET) \
  --gcc-toolchain=$(VORTEX_TOOLDIR)/$(GNU_DIR) \
  -march=$(MARCH) -mabi=$(MABI) \
  -Xclang -target-feature -Xclang +xvortex \
  -Xclang -target-feature -Xclang +zicond \
  -mllvm -disable-loop-idiom-all \
  -Wno-unused-command-line-argument -Wno-override-module \
  -O3 -mcmodel=medany -nostartfiles -nostdlib -fuse-ld=lld \
  -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections -ffp-contract=off \
  -D__VORTEX__ -DNDEBUG \
  -I. -I$(VORTEX_HOME)/sw/gfx -I$(VORTEX_HOME)/sw/kernel/include \
  -I$(VORTEX_HOME)/sw/common -I$(VORTEX_BUILD)/sw -I$(VORTEX_BUILD)/hw

VX_LDFLAGS := -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/sw/kernel/scripts/$(LINKER),--defsym=STARTUP_ADDR=0x80000000 \
  $(VORTEX_BUILD)/sw/kernel/libvortex2.a \
  -L$(VORTEX_TOOLDIR)/$(LIBC)/lib -lm -lc \
  $(VORTEX_TOOLDIR)/$(LIBCRT)/lib/baremetal/$(CRT_A)

all: $(OUTDIR)/gfx_frontend.vxbin

# Kernel sources are owned by the Vortex SDK (sw/gfx + sw/common), not copied
# here — this driver compiles them, the SDK is the single source of truth.
GFX_HDRS := $(wildcard $(VORTEX_HOME)/sw/gfx/*.h) $(VORTEX_HOME)/sw/common/gfx_frontend_abi.h

$(OUTDIR)/gfx_frontend.elf: gfx_frontend_kernel.cpp $(GFX_HDRS)
	$(CLANG) $(VX_CFLAGS) gfx_frontend_kernel.cpp $(VX_LDFLAGS) -o $@

$(OUTDIR)/gfx_frontend.vxbin: $(OUTDIR)/gfx_frontend.elf
	OBJCOPY=$(OBJCOPY) python3 $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@

clean:
	rm -f $(OUTDIR)/gfx_frontend.elf $(OUTDIR)/gfx_frontend.vxbin

.PHONY: all clean
