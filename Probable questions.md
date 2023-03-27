Q. What are natural rdinates and what are relative coordinates ? #definition
	Natural coordinates refer to postion and orientation of a an object with refrence to a fixed frame. It can be either the global coordinates or the base of the ronot. The natural cordinates will not change with motion of the robot.  
	Relative Coordinate/ Local Cordiante of the system refers to the postion and the orientation of object or part of robot in reference to local coordinate of the sytem. It is useful in decribing the postion the position of a joint with respect to parent joint. #definition 

Q. Explain the Recursive Kinematic Algo.
	#explain (imagine the 3 linked body) to compute $$ J_{3} , J_{2} $$ has to be calculated. For $$ J_{2} , J_{1} $$ has to be calculated. 
	Therefore the mass matrix
<<<<<<< HEAD
	$$ M = m_{1}.(J_{s1})^{T} + m_{2}.(J_{s2})^{T} + m_{2}.(J_{s2})^{T} $$
=======
	$$ M = m_{1}.(J_{s1})^{T}. J_{s1} + $$
>>>>>>> d76971e64ce4f3054162f9174148afb5aa73b3af
