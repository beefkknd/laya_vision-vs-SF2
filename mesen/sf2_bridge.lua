-- sf2_bridge.lua: lets Python drive SNES Street Fighter II running in Mesen 2.
--
-- Load it once in Mesen (Debug > Script Window > Open > Run, or `Mesen <rom> sf2_bridge.lua`). In Script Window >
-- Settings > Restrictions tick "Allow network access". It connects to whichever scripts/*.py is listening on
-- 127.0.0.1:47800 and reconnects on its own after each one finishes, so Mesen can stay open all day.
--
-- The emulator never runs ahead of Python: at every input poll this script either applies the next planned
-- input or, when the plan is used up, reports and blocks on the socket until Python sends the next command.
-- All file I/O (savestates, screenshots, logs) happens on the Python side.
--
-- Commands (one line each, from Python):
--   VARS <n>                   + n lines "name addr size signed"  (addr = offset into WRAM, 7E0000 -> 0)
--   RUN <n> <caps> <b1> .. <bn>   apply n frames of input; bi = "down+right+l" or "-"; caps = "0,8,12" or "-"
--   WATCH <n> <every>          n frames of player-controlled input; log it; screenshot every <every> frames
--   LOADSTATE <len>            + len bytes of a savestate made by SAVESTATE
--   SAVESTATE                  save now; bytes come back in the report
--   RESET                      reset the cartridge and report its new initial state
--   CAPTURE png|raw            screenshots as PNG (takeScreenshot) or raw RGB from the screen buffer; use raw when
--                              PNGs come back blank (headless --testrunner runs)
--   KEEP                       keep a headless test-runner alive during slow Python inference; no response
--   DUMP                       whole 128 KiB WRAM
--   QUIT                       disconnect, keep emulating, wait for the next Python run
--   EXIT                       end the Mesen process (headless --testrunner runs)
-- F9 while a WATCH is running saves a savestate; it comes back in that WATCH's report.
-- Report (to Python), after every command:
--   OBS <nrams> <ninputs> <nimgs> <statelen>, then nrams csv lines (RAM before frame 0..n), ninputs lines of
--   pressed buttons, nimgs x ("IMG <frame> <len>" + png bytes, or "RAW <frame> <w> <h> <len>" + RGB bytes),
--   then <statelen> savestate bytes.

local HOST, PORT = "127.0.0.1", 47800
do  -- SF2_BRIDGE_PORT overrides the port (needs "Allow access to I/O and OS functions"; ignored otherwise)
  local ok, v = pcall(function() return os.getenv("SF2_BRIDGE_PORT") end)
  if ok and tonumber(v or "") then PORT = tonumber(v) end
end
local BUTTONS = {"a", "b", "x", "y", "l", "r", "up", "down", "left", "right", "select", "start"}
local WRAM = emu.memType.snesWorkRam

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

local function send(s)
  local i = 1
  while i <= #s do
    local last, e, partial = conn:send(s, i)
    if last then i = last + 1
    elseif e == "timeout" then i = partial + 1
    else error("sf2_bridge: socket " .. tostring(e)) end
  end
end

local function recvLine()
  local line, e = conn:receive("*l")
  if not line then error("sf2_bridge: socket " .. tostring(e)) end
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

local vars = {}
local mode, n, k = nil, 0, 0          -- mode: nil (waiting) | "run" | "watch" | "busy" (savestate pending)
local plan, caps, every = {}, {}, 0
local rams, inputs, imgs, state = {}, {}, {}, nil
local cbRef, f9Down = nil, false
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

local function readVars()
  local t = {}
  for i, v in ipairs(vars) do
    if v.size == 1 then t[i] = emu.read(v.addr, WRAM, v.signed)
    elseif v.size == 2 then t[i] = emu.read16(v.addr, WRAM, v.signed)
    else t[i] = emu.read32(v.addr, WRAM, v.signed) end
  end
  return table.concat(t, ",")
end

local function pressed()
  local inp = emu.getInput(0)
  local t = {}
  for _, b in ipairs(BUTTONS) do if inp[b] then t[#t + 1] = b end end
  return #t > 0 and table.concat(t, "+") or "-"
end

local function inputTable(spec)
  local t = {}
  for _, b in ipairs(BUTTONS) do t[b] = false end
  if spec ~= "-" then for _, b in ipairs(split(spec, "+")) do t[b] = true end end
  return t
end

local function report()
  local nimg = 0
  for _ in pairs(imgs) do nimg = nimg + 1 end
  send(string.format("OBS %d %d %d %d\n", #rams, #inputs, nimg, state and #state or 0))
  for _, line in ipairs(rams) do send(line .. "\n") end
  for _, line in ipairs(inputs) do send(line .. "\n") end
  for idx, img in pairs(imgs) do
    if type(img) == "string" then
      send(string.format("IMG %d %d\n", idx, #img))
      send(img)
    else
      send(string.format("RAW %d %d %d %d\n", idx, img[1], img[2], #img[3]))
      send(img[3])
    end
  end
  if state then send(state) end
  rams, inputs, imgs, state = {}, {}, {}, nil
end

-- savestates can only be created / loaded from inside an "exec" memory callback.
-- keepMode: an F9 save while the player is recording; the watch chunk carries on and reports the state at its end.
local function armExec(pendingLoad, keepMode)
  cbRef = emu.addMemoryCallback(function()
    if pendingLoad then emu.loadSavestate(pendingLoad) else state = emu.createSavestate() end
    emu.removeMemoryCallback(cbRef, emu.callbackType.exec, 0, 0xFFFFFF)
    cbRef = nil
    if not keepMode then mode, n, k, caps = "run", 0, 0, {[0] = true} end  -- next poll reports where it landed
  end, emu.callbackType.exec, 0, 0xFFFFFF)
  if not keepMode then mode = "busy" end
end

local function serve()
  while true do
    local cmd = split(recvLine(), " ")
    local op = cmd[1]
    if op == "VARS" then
      vars = {}
      for i = 1, tonumber(cmd[2]) do
        local f = split(recvLine(), " ")
        vars[i] = {addr = tonumber(f[2]), size = tonumber(f[3]), signed = f[4] == "1"}
      end
      send("OK\n")
    elseif op == "RUN" then
      n, k, plan, caps = tonumber(cmd[2]), 0, {}, {}
      if cmd[3] ~= "-" then for _, c in ipairs(split(cmd[3], ",")) do caps[tonumber(c)] = true end end
      for i = 1, n do plan[i] = cmd[3 + i] end
      mode = "run"
      return
    elseif op == "WATCH" then
      n, k, every, caps = tonumber(cmd[2]), 0, tonumber(cmd[3]), {}
      mode = "watch"
      return
    elseif op == "LOADSTATE" then
      local data = conn:receive(tonumber(cmd[2]))
      armExec(data)
      return
    elseif op == "SAVESTATE" then
      armExec(nil)
      return
    elseif op == "RESET" then
      emu.reset()
      mode, n, k, caps = "run", 0, 0, {[0] = true}
      return
    elseif op == "CAPTURE" then
      capture = cmd[2] == "raw" and "raw" or "png"
      send("OK\n")
    elseif op == "KEEP" then
      -- Deliberately silent: Python will next expect an OBS response, not a heartbeat response.
    elseif op == "DUMP" then
      local size = emu.getMemorySize(WRAM)
      local parts, chunk = {}, {}
      for a = 0, size - 1 do
        chunk[#chunk + 1] = emu.read(a, WRAM, false)
        if #chunk == 4096 then parts[#parts + 1] = string.char(table.unpack(chunk)); chunk = {} end
      end
      if #chunk > 0 then parts[#parts + 1] = string.char(table.unpack(chunk)) end
      local blob = table.concat(parts)
      send("BIN " .. #blob .. "\n")
      send(blob)
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
      send("ERR unknown command " .. tostring(op) .. "\n")
    end
  end
end

local function onPoll()
  polls = polls + 1
  if conn == nil then
    if polls - lastTry < 60 then return end
    lastTry = polls
    if not tryConnect() then return end
    emu.displayMessage("sf2_bridge", "connected to Python on port " .. PORT)
    hello()
  end
  while true do
    if mode == "busy" then return end               -- waiting for the exec callback
    if mode == nil then
      serve()
      if conn == nil or mode == nil or mode == "busy" then return end
    end
    -- index k: state before frame k (k == n: after the last one)
    rams[#rams + 1] = readVars()
    if caps[k] or (mode == "watch" and every > 0 and k % every == 0 and k < n) then
      imgs[k] = grab()
    end
    if mode == "watch" and k < n then
      inputs[#inputs + 1] = pressed()
      local f9 = emu.isKeyPressed("F9")
      if f9 and not f9Down and not state and not cbRef then armExec(nil, true) end
      f9Down = f9
    end
    if k >= n then
      report()
      mode = nil
    else
      if mode == "run" then emu.setInput(inputTable(plan[k + 1]), 0) end
      k = k + 1
      return
    end
  end
end

emu.addEventCallback(onPoll, emu.eventType.inputPolled)
emu.displayMessage("sf2_bridge", "loaded; waiting for a Python script on port " .. PORT)
