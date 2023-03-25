## Constrained Optimization Problem 
[[Constained Problem]] involves finding the maximum oe minimum values of a function subject to a set constrains set to its variables.  
let  $$ f(x,y)$$ be function and $$ g(x,y) \hspace{1 cm} h(x,y) $$ be two contrains imposed on it.
We can repersent this problem with Lagrange Multipliers $$ f(x,t, \lambda, \mu) = f(x,y) + \lambda(x,y) + \mu(x,y) $$
where $$ f(x,t, \lambda, \mu) $$ is called agumented Lagrangian and combines the objective function with constrains. 
### The solution
- The conditions for Optimality are given by [[Karush-Kuhn-Tucker/KKT]] conditions: [[Conditions for Optimality]]
	- [[Stationarity]] : The gradient of the Lagrangian w.r.t to x and y are zero. $$ \frac{dL}{dx} = 0 \hspace{1 cm} \frac{dL}{dy} =0 $$
	- [[Primal Feasibility]] : The constains are satisfied $$ g(x,y) = 0 \hspace{1 cm} f(x,y) = 0 $$
	- [[Dual Feasibility]] : The Lagrangian multipliers are non negative : $$ \lambda = 0 \hspace{1 cm} \mu =0 $$
	- [[Complementary slackness]] : The Langrangian multipliers and constrains satisfy the complementray slackness condition: $$ \lambda \times g(x,y) = 0 \hspace{1 cm} and \hspace{1 cm} \mu \times h(x,y) = 0 $$
- Solving the constrained optimization involves finding the values of $$ x \hspace{1cm},y \hspace{1cm},\lambda \hspace{1cm} \mu $$
	that satisfy the [[Karush-Kuhn-Tucker/KKT]] conditions. This can be done using [[Numerical Optimization Methods]]


## IK Problem, Steps involved in solving it. 
	1. Define the robot structure, including the number of joints and their limits.
	2. Specify the end effector pose. 
	3. Formulate the IK problem ==> nonlinear optimization problem. Where the objective is to minimize the distance between the computed end effector pose and desired end effector pose.
	4. Computen the Jacobian matrix. Jacobian relates changes in joint angles/postions to the change in end effecot pose. Jacobian is iteratively solve the inverse kinematic problem.
	5. Chose the Optimization method; eg: Gradient based methods.
	6. Iterate to convergence. IK is solved by iteratively by updating the joint and angles/posistions in steepest descent of the objecttive function. This continues until the convergence criteria is met. 
	7. Verfiy the solution. 

## ID Problem, Steps involved in solvin it.
Goal: Finding joint torques or forces that are required to produce desired motion or tragetory. 

		1.Define the kinematic and dynamic model of the robotic system.
		2.Specifiy the desired motion targectory.
		3.Compute joint velocites and acceleration using the kinematic model. 
		4.Compute the joint torques or forces using the dynamic model
		5.Account for external forces 
		6. Implementation of control.

## Projected Newton Euler Equation
- Model the motion and dynamics of a rigid body system
- predict the motion of the system and control its behavior in real-time
- The equations take the following form:
	- Linear acceleration Equation : -  relates to external foces and torques to linear acceleartion of center of mass 
	- Angluar accleration Equation
	- Force Projected Equation
	- Torque Projected Equation