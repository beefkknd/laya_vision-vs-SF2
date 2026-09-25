-- Runs mesen/sf2_bridge.lua outside Mesen: a mock of the `emu` API with a toy two-fighter game in 128 KiB of
-- "WRAM" (struct stride 0x200, life at +0x30, x at +0x22, y at +0x26). Used by tests/test_lua_bridge.py.
--   lua5.4 tests/lua/mock_mesen.lua <path to sf2_bridge.lua> <png file used for every screenshot>
local bridgePath, pngPath = arg[1], arg[2]
local png = assert(io.open(pngPath, "rb")):read("a")

local W = {}
for i = 0, 0x1FFFF do W[i] = 0 end
local function w16(a, v) if v < 0 then v = v + 0x10000 end W[a] = v & 0xFF; W[a + 1] = (v >> 8) & 0xFF end
local function r16(a, signed) local v = W[a] | (W[a + 1] << 8); if signed and v >= 0x8000 then v = v - 0x10000 end return v end

local g
local function reset() g = {t = 0, hp = {144, 144}, x = {80, 176}, y = {200, 200}, jt = 0, cd = 0, timer = 99, ko = 0} end
local function store()
  for p = 1, 2 do
    local base = 0x500 + (p - 1) * 0x200
    w16(base + 0x30, g.hp[p]); w16(base + 0x22, g.x[p]); w16(base + 0x26, g.y[p])
  end
  W[0x100] = g.timer
end
reset(); store()

local input = {}
local function step()
  g.t = g.t + 1
  if g.ko > 0 then                                -- KO screen: life stays at 0, then both bars refill
    g.ko = g.ko - 1
    if g.ko == 0 then g.hp = {144, 144}; g.x = {80, 176}; g.y = {200, 200}; g.jt = 0 end
    store()
    return
  end
  if input.right then g.x[1] = g.x[1] + 2 end
  if input.left then g.x[1] = g.x[1] - 2 end
  if input.up and g.jt == 0 then g.jt = 30 end
  if g.jt > 0 then g.jt = g.jt - 1; g.y[1] = 200 - (225 - (g.jt - 15) ^ 2) // 4 end
  if g.cd > 0 then g.cd = g.cd - 1 end
  if input.l and g.cd == 0 and math.abs(g.x[2] - g.x[1]) < 70 then g.hp[2] = g.hp[2] - 4; g.cd = 20 end
  if g.t % 120 == 0 then g.hp[1] = g.hp[1] - 9 end
  if g.t % 60 == 0 then g.timer = g.timer - 1 end
  if g.hp[1] <= 0 or g.hp[2] <= 0 then            -- like SF2 World Warrior: life clamps at 0 on a KO
    g.hp[1], g.hp[2] = math.max(0, g.hp[1]), math.max(0, g.hp[2]); g.ko = 150
  end
  store()
end

local polls, execs, stopped = {}, {}, false
local nextRef = 1
emu = {
  memType = {snesWorkRam = 1},
  eventType = {inputPolled = 1},
  callbackType = {exec = 1},
  getRomInfo = function() return {name = "Mock Fighter", fileSha1Hash = "mocksha1"} end,
  read = function(a) return W[a] end,
  read16 = function(a, _, signed) return r16(a, signed) end,
  read32 = function(a) return r16(a) | (r16(a + 2) << 16) end,
  getMemorySize = function() return 0x20000 end,
  getInput = function() return input end,
  setInput = function(t) input = t end,
  isKeyPressed = function() return false end,
  takeScreenshot = function() return png end,      -- a constant image: looks "blank" to capture=auto
  getScreenSize = function() return {width = 256, height = 224} end,
  getScreenBuffer = function()                     -- raw ARGB, a white column at my x
    local buf = {}
    for yy = 0, 223 do
      for xx = 0, 255 do buf[yy * 256 + xx + 1] = (xx == g.x[1] % 256) and 0xFFFFFFFF or 0xFF102030 end
    end
    return buf
  end,
  addEventCallback = function(fn) polls[#polls + 1] = fn end,
  addMemoryCallback = function(fn) local r = nextRef; nextRef = r + 1; execs[r] = fn; return r end,
  removeMemoryCallback = function(r) execs[r] = nil end,
  createSavestate = function() return string.format("%d,%d,%d,%d,%d,%d,%d", g.t, g.hp[1], g.hp[2], g.x[1], g.x[2], g.y[1], g.y[2]) end,
  loadSavestate = function(s)
    local v = {}
    for n in string.gmatch(s, "-?%d+") do v[#v + 1] = tonumber(n) end
    reset(); g.t, g.hp[1], g.hp[2], g.x[1], g.x[2], g.y[1], g.y[2] = table.unpack(v); store()
  end,
  displayMessage = function() end,
  log = function(s) io.stderr:write(s .. "\n") end,
  stop = function() stopped = true end,
}

dofile(bridgePath)
for _ = 1, 200000 do
  input = {}                               -- like Mesen: setInput lasts until the next poll
  for _, fn in ipairs(polls) do fn() end
  if stopped then break end
  for r, fn in pairs(execs) do fn() end   -- "the next instruction executes"
  step()
end
