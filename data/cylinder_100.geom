# Cylinder, diameter 1, with 100 points
# y=-2.0: equivalent to training data offset (cylinder at y=0 in domain centered at y=2)
# With -yoffset -4 (domain center y=0), this gives -2.0 offset from domain center

body Cylinder
  circle_n 0 -2.0 0.5 100
end
