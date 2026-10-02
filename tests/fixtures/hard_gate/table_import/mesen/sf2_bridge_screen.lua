-- screen-only bridge: no emu.read( on snesWorkRam here, comments do not count
--[[ a block comment: emu.read(0x0D18, emu.memType.snesWorkRam, false) ]]
local size = emu.getScreenSize()
emu.setInput({}, 0, 0, 0)
