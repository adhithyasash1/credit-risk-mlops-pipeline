-- Each request hits /predict/{random application_id 0..999}
math.randomseed(os.time())
request = function()
  local id = math.random(0, 999)
  return wrk.format("GET", "/predict/" .. id)
end
