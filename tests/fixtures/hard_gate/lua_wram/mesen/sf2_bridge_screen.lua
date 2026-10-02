-- screen-only bridge, seeded with a work-RAM read
local WRAM = emu.memType.snesWorkRam
local x = emu.read16(0x0D18, WRAM, false)
