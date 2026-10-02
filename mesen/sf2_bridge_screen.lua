-- sf2_bridge_screen.lua: the SCREEN-ONLY bridge for screen-only play (docs/laya_text_only_plan.md, "The rule: no RAM
-- in play"). A copy of sf2_bridge.lua with every RAM path removed: no VARS, no DUMP, no POKE, no RAM rows in a report.
-- During play the emulator gives screen frames and takes buttons, nothing else (plus loading the start savestate).
--
-- Commands (one line each, from Python):
--   RUN <n> <caps> <b1> .. <bn>   apply n frames of input; bi = "down+right+l" or "-"; caps = "0,8,12" or "-"
--   LOADSTATE <len>            + len bytes of a savestate (the run's start state)
--   CAPTURE png|raw            screenshots as PNG or raw RGB from the screen buffer (raw for headless runs)
--   KEEP                       keep a headless test-runner alive during slow Python inference; no response
--   QUIT / EXIT                disconnect / end the Mesen process
--   anything else (VARS, DUMP, POKE, SAVESTATE, WATCH, RESET, ...) -> "ERR ram-free bridge: <op>"
-- Report (to Python), after RUN / LOADSTATE:
--   OBS 0 0 <nimgs> 0, then nimgs x ("IMG <frame> <len>" + png bytes, or "RAW <frame> <w> <h> <len>" + RGB bytes).
--   The RAM row count is always 0: this script never reads WRAM.

local HOST, PORT = "127.0.0.1", 47800
-- Headless copies (sf2/headless.py) set this: when Python goes away (exits, crashes, is killed) the socket closes and
-- the test runner ends instead of emulating on for its whole --timeout.
local EXIT_ON_DISCONNECT = false
do  -- SF2_BRIDGE_PORT overrides the port (needs "Allow access to I/O and OS functions"; ignored otherwise)
  local ok, v = pcall(function() return os.getenv("SF2_BRIDGE_PORT") end)
  if ok and tonumber(v or "") then PORT = tonumber(v) end
end
local BUTTONS = {"a", "b", "x", "y", "l", "r", "up", "down", "left", "right", "select", "start"}

local socket = require("socket.core")
local conn = nil
local lastTry = -1000
local polls = 0

-- Stays loaded between Python runs: when no Python script is listening it retries once a second, so you can
-- leave Mesen open and just start the next scripts/*.py.
local function tryConnect()
  local c = socket.tcp()
  c:settimeout(0.05)
  local ok = c:connect(HOST, PORT)
  if not ok then c:close(); return false end
  c:settimeout(nil)
  c:setoption("tcp-nodelay", true)
  conn = c
  return true
end

local function lost(e)
  if EXIT_ON_DISCONNECT then emu.stop(1) end
  error("sf2_bridge: socket " .. tostring(e))
end

local function send(s)
  local i = 1
  while i <= #s do
    local last, e, partial = conn:send(s, i)
    if last then i = last + 1
    elseif e == "timeout" then i = partial + 1
    else lost(e) end
  end
end

local function recvLine()
  local line, e = conn:receive("*l")
  if not line then lost(e) end
  return line
end

local function split(s, sep)
  local out = {}
  for part in string.gmatch(s, "([^" .. sep .. "]+)") do out[#out + 1] = part end
  return out
end

local function hello()
  local rom = emu.getRomInfo()
  local name = string.gsub(rom.name or "?", "%s", "_")
  send("HELLO " .. (rom.fileSha1Hash or "?") .. " " .. name .. "\n")
end

local mode, n, k = nil, 0, 0          -- mode: nil (waiting) | "run" | "busy" (savestate load pending)
local plan, caps = {}, {}
local imgs = {}
local cbRef = nil
local capture = "png"

-- one screenshot: a PNG string, or {w, h, rgb bytes} from the raw ARGB screen buffer
local function grab()
  if capture == "png" then return emu.takeScreenshot() end
  local size = emu.getScreenSize()
  local buf = emu.getScreenBuffer()
  local parts, chunk = {}, {}
  for i = 1, #buf do
    local c = buf[i]
    chunk[#chunk + 1] = (c >> 16) & 0xFF
    chunk[#chunk + 1] = (c >> 8) & 0xFF
    chunk[#chunk + 1] = c & 0xFF
    if #chunk >= 3072 then parts[#parts + 1] = string.char(table.unpack(chunk)); chunk = {} end
  end
  if #chunk > 0 then parts[#parts + 1] = string.char(table.unpack(chunk)) end
  return {size.width, size.height, table.concat(parts)}
end

local function inputTable(spec)
  local t = {}
  for _, b in ipairs(BUTTONS) do t[b] = false end
  if spec ~= "-" then for _, b in ipairs(split(spec, "+")) do t[b] = true end end
  return t
end

-- Mesen 2's setInput reads port and subport from the top of a 4-slot stack (lua_settop 4), so with the documented
-- setInput(input, port) the port always comes out 0. Passing 4 values puts port in slot 3, subport in slot 4.
local function setPad(t, port) emu.setInput(t, port, port, 0) end

-- "p1" drives controller 1 only; "p1/p2" drives both (controller 2 is left alone otherwise)
local function applyFrame(spec)
  local slash = string.find(spec, "/", 1, true)
  if not slash then setPad(inputTable(spec), 0); return end
  setPad(inputTable(string.sub(spec, 1, slash - 1)), 0)
  setPad(inputTable(string.sub(spec, slash + 1)), 1)
end

local function report()
  local nimg = 0
  for _ in pairs(imgs) do nimg = nimg + 1 end
  send(string.format("OBS 0 0 %d 0\n", nimg))
  for idx, img in pairs(imgs) do
    if type(img) == "string" then
      send(string.format("IMG %d %d\n", idx, #img))
      send(img)
    else
      send(string.format("RAW %d %d %d %d\n", idx, img[1], img[2], #img[3]))
      send(img[3])
    end
  end
  imgs = {}
end

-- savestates can only be loaded from inside an "exec" memory callback; the next poll reports where it landed.
local function armLoad(data)
  cbRef = emu.addMemoryCallback(function()
    emu.loadSavestate(data)
    emu.removeMemoryCallback(cbRef, emu.callbackType.exec, 0, 0xFFFFFF)
    cbRef = nil
    mode, n, k, caps = "run", 0, 0, {[0] = true}
  end, emu.callbackType.exec, 0, 0xFFFFFF)
  mode = "busy"
end

local function serve()
  while true do
    local cmd = split(recvLine(), " ")
    local op = cmd[1]
    if op == "RUN" then
      n, k, plan, caps = tonumber(cmd[2]), 0, {}, {}
      if cmd[3] ~= "-" then for _, c in ipairs(split(cmd[3], ",")) do caps[tonumber(c)] = true end end
      for i = 1, n do plan[i] = cmd[3 + i] end
      mode = "run"
      return
    elseif op == "LOADSTATE" then
      local data, e = conn:receive(tonumber(cmd[2]))
      if not data then lost(e) end
      armLoad(data)
      return
    elseif op == "CAPTURE" then
      capture = cmd[2] == "raw" and "raw" or "png"
      send("OK\n")
    elseif op == "KEEP" then
      -- Deliberately silent: Python will next expect an OBS response, not a heartbeat response.
    elseif op == "PING" then
      send("PONG\n")
    elseif op == "QUIT" then                    -- Python is done; wait for the next one
      conn:close()
      conn, mode = nil, nil
      return
    elseif op == "EXIT" then                    -- headless (--testrunner) runs: end the Mesen process
      conn:close()
      emu.stop(0)
      return
    else
      send("ERR ram-free bridge: " .. tostring(op) .. "\n")
    end
  end
end

local function onPoll()
  polls = polls + 1
  if conn == nil then
    if polls - lastTry < 60 then return end
    lastTry = polls
    if not tryConnect() then return end
    emu.displayMessage("sf2_bridge_screen", "connected to Python on port " .. PORT)
    hello()
  end
  while true do
    if mode == "busy" then return end               -- waiting for the exec callback
    if mode == nil then
      serve()
      if conn == nil or mode == nil or mode == "busy" then return end
    end
    -- index k: the screen before frame k (k == n: after the last one)
    if caps[k] then imgs[k] = grab() end
    if k >= n then
      report()
      mode = nil
    else
      applyFrame(plan[k + 1])
      k = k + 1
      return
    end
  end
end

emu.addEventCallback(onPoll, emu.eventType.inputPolled)
emu.displayMessage("sf2_bridge_screen", "loaded (screen only, no RAM); waiting for a Python script on port " .. PORT)
