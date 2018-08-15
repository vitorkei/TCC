--Converts HSL in (degrees, percent, percent) to (0-255) value
local function stdv(h,s,l,a)
  local sh, ss, sl

  sh = h*255/360
  ss = s*255/100
  sl = l*255/100
  a = a or 255

  return sh, ss, sl, a
end

--Our constants
local ori_color = {244, 107, 66}
local ori_hsl = {stdv(14,89,61)}
local target_color = {95, 66, 244}
local target_hsl = {stdv(250,89,61)}
local radius = 40

--Converts HSL to RGB. (input and output range: 0 - 255)
local function convert(h, s, l, a)
	if s<=0 then return l,l,l,a end
	h, s, l = h/256*6, s/255, l/255
	local c = (1-math.abs(2*l-1))*s
	local x = (1-math.abs(h%2-1))*c
	local m,r,g,b = (l-.5*c), 0,0,0
	if h < 1     then r,g,b = c,x,0
	elseif h < 2 then r,g,b = x,c,0
	elseif h < 3 then r,g,b = 0,c,x
	elseif h < 4 then r,g,b = 0,x,c
	elseif h < 5 then r,g,b = x,0,c
	else              r,g,b = c,0,x
	end return (r+m)*255,(g+m)*255,(b+m)*255,a
end

-------------------------------------

function love.load()
  love.graphics.setBackgroundColor(255, 255, 255)
  love.graphics.setFont(love.graphics.newFont(30))
end

function love.draw()
  local font = love.graphics.getFont() --Font used
  local width = love.graphics.getWidth() --Width of screen

  local initial_pos = 100
  local target_pos = width - initial_pos

  --Draw original and target colors
  local text = "Cor Inicial"
  love.graphics.setColor(0, 0, 0)
  love.graphics.print(text, initial_pos - font:getWidth(text)/2, 20)
  local x, y = initial_pos, 100
	love.graphics.setColor(unpack(ori_color))
	love.graphics.circle("fill", x, y, radius)

  local text = "Cor Final"
  love.graphics.setColor(0, 0, 0)
  love.graphics.print(text, target_pos - font:getWidth(text)/2, 20)
  love.graphics.setColor(unpack(target_color))
  x = target_pos
  love.graphics.circle("fill", x, y, radius)

  --Draw rgb transition
  local text = "Transição em RGB"
  love.graphics.setColor(0, 0, 0)
  love.graphics.print(text, width/2 - font:getWidth(text)/2, 200)

  x, y = initial_pos, 300
  local max = 100
  local step = (target_pos-initial_pos)/max

  for i = 0, max do
    local r = ori_color[1] * (1-(i/max)) + target_color[1] * (i/max)
    local g = ori_color[2] * (1-(i/max)) + target_color[2] * (i/max)
    local b = ori_color[3] * (1-(i/max)) + target_color[3] * (i/max)

    love.graphics.setColor(r, g, b)
    love.graphics.circle("fill", x, y, radius)
    x = x + step

  end

  --Draw hsl transition
  local text = "Transição em HSL"
  love.graphics.setColor(0, 0, 0)
  love.graphics.print(text, width/2 - font:getWidth(text)/2, 400)

  x, y = initial_pos, 500
  local max = 100
  local step = (target_pos-initial_pos)/max

  for i = 0, max do
    local r = ori_hsl[1] * (1-(i/max)) + target_hsl[1] * (i/max)
    local g = ori_hsl[2] * (1-(i/max)) + target_hsl[2] * (i/max)
    local b = ori_hsl[3] * (1-(i/max)) + target_hsl[3] * (i/max)

    love.graphics.setColor(convert(r, g, b))
    love.graphics.circle("fill", x, y, radius)
    x = x + step

  end


end
