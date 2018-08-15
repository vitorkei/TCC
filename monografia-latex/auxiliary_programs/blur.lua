love.window.setMode(900, 500)

--Shaders for blur effect
Horizontal_Blur_Shader = love.graphics.newShader[[
  extern number win_width; //1 / (screen width)
  const float kernel[5] = float[](0.270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);
  vec4 effect(vec4 color, sampler2D tex, vec2 tex_coords, vec2 pos) {
    color = texture2D(tex, tex_coords) * kernel[0];
    for(int i = 1; i < 5; i++) {
      color += texture2D(tex, vec2(tex_coords.x + 6*i * win_width, tex_coords.y)) * kernel[i];
      color += texture2D(tex, vec2(tex_coords.x - 6*i * win_width, tex_coords.y)) * kernel[i];
    }
    return color;
  }
]]

Vertical_Blur_Shader = love.graphics.newShader[[
    extern number win_height; //1 / (screen height)
		const float kernel[5] = float[](0.270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);
		vec4 effect(vec4 color, sampler2D tex, vec2 tex_coords, vec2 pos) {
			color = texture2D(tex, tex_coords) * kernel[0];
			for(int i = 1; i < 5; i++) {
				color += texture2D(tex, vec2(tex_coords.x, tex_coords.y + 6*i * win_height)) * kernel[i];
				color += texture2D(tex, vec2(tex_coords.x, tex_coords.y - 6*i * win_height)) * kernel[i];
			}
			return color;
		}
]]

Horizontal_Blur_Shader:send("win_width", 1/love.graphics.getWidth())
Vertical_Blur_Shader:send("win_height", 1/love.graphics.getHeight())

image = love.graphics.newImage("linuxlogo.png")
sx = .1
sy = sx
canvas = love.graphics.newCanvas(love.graphics.getWidth(), love.graphics.getHeight())
-------------------------------------

function love.load()
  love.graphics.setBackgroundColor(255, 255, 255)
  love.graphics.setFont(love.graphics.newFont(25))
end

function drawObj(x, y)
  love.graphics.setColor(255, 255, 255)
  love.graphics.draw(image,x,y, nil, sx, sy)
end

function love.draw()
  local x, y = 30, 200
  local w = image:getWidth()*sx
  local text
  local font = love.graphics.getFont()

  --Normal object
  drawObj(x,y)
  text = "sem shader"
  love.graphics.setColor(0,0,0)
  love.graphics.printf(text, x, y - 40, w, "center")

  --Horizontal Blur Applied
  x = x + 200
  love.graphics.setShader(Horizontal_Blur_Shader)
  drawObj(x,y)
  love.graphics.setShader()
  text = "shader horizontal"
  love.graphics.setColor(0,0,0)
  love.graphics.printf(text, x, y - 60, w, "center")

  --Vertical Blur Applied
  x = x + 200
  love.graphics.setShader(Vertical_Blur_Shader)
  drawObj(x,y)
  love.graphics.setShader()
  text = "shader   vertical"
  love.graphics.setColor(0,0,0)
  love.graphics.printf(text, x, y - 60, w, "center")

  --Horizontal and Vertical Blur applied
  x = x + 200
  love.graphics.setCanvas(canvas)
  --  love.graphics.setColor(210, 210, 255)
--    love.graphics.rectangle("fill", x, y, love.graphics.getWidth(), love.graphics.getHeight())
    --love.graphics.setColor(255,255,255)
    love.graphics.setShader(Horizontal_Blur_Shader)
    drawObj(x,y)
    love.graphics.setShader()
  love.graphics.setCanvas()

  love.graphics.setShader(Vertical_Blur_Shader)
  love.graphics.setBlendMode("alpha", "premultiplied")
  love.graphics.draw(canvas)
  love.graphics.setBlendMode("alpha")
  love.graphics.setShader()
  text = "horizontal + vertical"
  love.graphics.setColor(0,0,0)
  love.graphics.printf(text, x, y - 70, w, "center")


end
