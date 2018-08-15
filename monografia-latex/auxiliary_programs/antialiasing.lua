
--Generic Smooth Circle Shader (for objects that change size a lot)
    Generic_Smooth_Circle_Shader = love.graphics.newShader[[
      vec4 effect( vec4 color, Image texture, vec2 texture_coords, vec2 screen_coords ){
        vec4 pixel = Texel(texture, texture_coords );//This is the current pixel color
        vec2 center = vec2(0.5,0.5);
        pixel.a = 1 - smoothstep(.49, .5, distance(center, texture_coords));
        return pixel * color;
      }
    ]]

PIXEL = love.graphics.newImage("pixel.png") -- Pixel for shaders

function Draw_Smooth_Circle(x, y, r)
    x = x - r
    y = y - r

    --Draw the circle
    love.graphics.setShader(Generic_Smooth_Circle_Shader)
    love.graphics.draw(PIXEL, x, y, 0, 2*r)
    love.graphics.setShader()


end


function love.draw()
	local x, y, r = 200, 300, 6
	love.graphics.setBackgroundColor(255, 255, 255)

	love.graphics.setColor(66, 155, 244)
	love.graphics.circle("fill", x, y, r)

	Draw_Smooth_Circle(x+3*r,y,r)


end
