# Phase field dendritic crystal growth

We are interested in numerically solving the following anisotropic phase-field dendritic crystal growth model equations in the domain  
$\Omega \subset \mathbb{R}^2$:

$$\rho(\phi)\phi_t = -\frac{\delta E}{\delta \phi} - \frac{\lambda}{\varepsilon} h'(\phi)T, \tag{2.1}$$

$$
T_t = \nabla \cdot (D\nabla T) + K h'(\phi)\phi_t, \tag{2.2}
$$

where $\phi(x,t)$ is the phase function to label the liquid and solid phases,  
$\rho(\phi)>0$ is the mobility parameter that can be chosen either as a constant [39],  
or as a function of $\phi$ [35].  
$\varepsilon > 0$ is a parameter used to control the interface width,  
$\lambda$ is the linear kinetic coefficient.  
In Equation (2.2), $T(x,t)$ is the scaled temperature,  
$D$ is the constant diffusion rate of the temperature,  
and $K$ is the latent heat parameter that controls the speed of heat transfer along with the interface.  
It is worth noting that the efficiency of the schemes we propose below covers the case $D$ is a function of $\phi$ [1]; see Remark 3.3.  
The function $h(\phi)$ is defined by

$$
h(\phi) := \frac{1}{5}\phi^5 - \frac{2}{3}\phi^3 + \phi,
$$

which represents a generation of latent heat.  
Following the phenomenological free energy used in [14], we consider here

$$
E(\phi, T) = \int_{\Omega} \left( 
\frac{1}{2}\kappa^2(\nabla\phi)|\nabla\phi|^2
+ \frac{1}{\varepsilon^2}F(\phi)
+ \frac{\lambda}{2\varepsilon K} T^2
\right)\, dx, \tag{2.3}
$$

where $F(\phi) = \frac14 (\phi^2 - 1)^2$ is the double-well type Ginzburg–Landau potential.  
$\kappa(\cdot)$ in (2.3) is a function describing the anisotropic property, which takes the form [14,17]:

$$
\kappa(\nabla\phi) = 1 + \sigma\cos(m\theta), \tag{2.4}
$$

where $m$ is a model number of anisotropy,  
$\sigma$ is the parameter for the anisotropy strength,  
and $\theta = \arctan\left(\frac{\phi_y}{\phi_x}\right)$.  
The variational derivative of $E$ with respect to $\phi$ is:

$$
\frac{\delta E}{\delta\phi}
= -\nabla \cdot \left( 
\kappa^2(\nabla\phi)\nabla\phi
+ \kappa(\nabla\phi)|\nabla\phi|^2 H(\phi)
\right)
+ \frac{f(\phi)}{\varepsilon^2},
$$

where $H(\phi)$ is the variational derivative of  
$\mathcal{K}(\phi):=\int_\Omega \kappa(\nabla\phi)dx$,  
and $f(\phi)=F'(\phi)$.  
In the case $m=4$, a direct calculation shows

$$
H(\phi) := \frac{\delta \mathcal{K}(\phi)}{\delta \phi}
= 4\sigma \frac{4}{|\nabla\phi|^6}
\left(
\phi_x(\phi_x^2\phi_y^2 - \phi_y^4),\,
\phi_y(\phi_x^2\phi_y^2 - \phi_x^4)
\right). \tag{2.5}
$$

For convenience, we only consider that the equations (2.1) and (2.2) are subject to the Neumann boundary conditions

$$
\frac{\partial \phi}{\partial n}\bigg|_{\partial\Omega} = 0, \qquad
\frac{\partial T}{\partial n}\bigg|_{\partial\Omega} = 0, \tag{2.6}
$$

although other boundary conditions such as periodic conditions are possible.
