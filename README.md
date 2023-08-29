# VanVleckCalculator
## 1. Introduction

VanVleckCalculator is a Python package for the calculation of the van Vleck distortion modes of octahedra, along with various other parameters including octahedral volume, angular shear, and various distortion parameters such as bond length distortion index, quadratic elongation, and the Van Vleck octahedral distortion modes.

It is developed by [Liam Nagle-Cocco](https://lnaglecocco.github.io), a PhD student in physics at the University of Cambridge. For questions or suggestions, email [lavn2@cam.ac.uk](lavn2@cam.ac.uk).

### 1.1 Requirements
- Python 3.8 or above
- numpy
- pymatgen

### 1.2 Setup guide

Download the van_vleck_calculator.py file, or better yet, close the repository using Git or GitHub Desktop so you can pull any updates.

Then, in a Jupyter notebook (or whatever you use to write Python), write the following:

```
import sys
sys.path.insert(1,r"C:\Users\User\Documents\GitHub\VanVleckCalculator\code")
from van_vleck_calculator import *
```
where the path should be the path containing VanVleckCalculator's code. This will enable VanVleckCalculator to be accessed.

## 2. Usage guide

For more in-depth information, see the documentation or the code itself. See also worked examples.

### 2.1 Creating a Pymatgen.core.structure object

To initialise an `Octahedron` object, it is necessary to first create a [Pymatgen Structure object](https://pymatgen.org/pymatgen.core.structure.html) for your unit cell. The easiest way to do this is by importing a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File). It is also possible to manually create a `Pymatgen.core.structure` object using known lattice parameters and atomic coordinates.

#### 2.1.1 Using a CIF

The following code can be used to generate a `Pymatgen.core.structure` object using a CIF file.

```
from pymatgen.core.structure import Structure
struc = Structure.from_file("cif_name.cif")
```

If you are working with a very large CIF file (such as from a molecular dynamics simulation or big box analysis of pair distribution function data), it is recommended that you set the optional `frac_tolerance` argument to be zero, to avoid shifting some atomic sites to high-symmetry positions. This requires Pymatgen version 2023.01.20 or later. For example:

```
from pymatgen.core.structure import Structure
struc = Structure.from_file("cif_name.cif",frac_tolerance=0)
```

This approach can also be used to import a VASP CONTCAR file, and several other crystal structure file types as documented on the Pymatgen website.

#### 2.1.2 Doing it manually

It is recommended you use a CIF, but if this is not feasible/desirable for your particular case, the following approach may be used. This example will generate a `pymatgen.core.structure` object for α-MnO<sub>2</sub> in the _I4/m_ space group.

```
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
latt = Lattice(matrix=([9.85,0,0],[0,9.85,0],[0,0,2.86]))
local_coords = (
    [0.35049,0.16700,0],
    [0.15137,0.19876,0],
    [0.54139,0.16782,0]
)
struc = Structure(lattice=latt,species=["Mn","O","O"],coords=local_coords)
struc = struc.from_spacegroup(lattice=latt,species=[Mn","O","O"],coords=local_coords,sg="I4/m")
```

For more information on this, see the [documentation for Pymatgen](https://pymatgen.org/pymatgen.core.structure.html).

### 2.2 Initialising an Octahedron object

To generate an `Octahedron` object, it is necessary to tell the code where to consider the "centre" of your octahedron and which `pymatgen.core.structure` object to search in. There are other optional arguments, giving the ability to forbid certain species from being ligands or mandate ligands can only be a certain species, giving the range over which to search for ligands, and other options.

The only two mandatory arguments are a `pymatgen.core.sites.PeriodicSite` object ([see here](https://pymatgen.org/pymatgen.core.sites.html)) and the `pymatgen.core.structure` object. Sites within a structure can be accessed by indexing, like an array. In the α-MnO<sub>2</sub> example given previously, an `Octahedron` object can easily be created for the MnO<sub>6</sub> octahedra as follows:

```
oct1 = Octahedron(struc[0],struc)
```

where `struc[0]` will refer to the `0`th site in `struc`, which in this case will be a Mn site. Here, default values will be used for the range over which to search for ligands, and any site within this range will be considered a ligand. Often, the default values will work well enough. If we want to manually set a maximum distance over which to search for ligands, and exclude Mn ions from having other Mn ions as ligands, we could write the following:

```
oct1 = Octahedron(struc[0],struc,forbidden_ligands=["Mn"],ligands_max_distance=2.5)
```

Now, only atoms that are not Mn can be assigned as ligands to the oct1 object, and only atoms less than 2.5 Ångstroms will be considered. This is fine for α-MnO<sub>2</sub>, but what if we have a larger number of elements? We can force O to be the ligand as follows:

```
oct1 = Octahedron(struc[0],struc,possible_ligands=["O"])
```

For the sake of argument, if we had a system containing many different species, and we wanted O and Cl as possible ligands, we could use the following:

```
oct1 = Octahedron(struc[0],struc,possible_ligands=["O","Cl"])
```

#### 2.3 Calculation of van Vleck modes and angular shear

##### 2.3.1 Summary

Van Vleck defined a set of six modes based on the symmetry of the octahedra. These are labelled Q<sub>1</sub> to Q<sub>6</sub>, where Q<sub>2</sub> and <sub>3</sub> are the modes sensitive to [Jahn-Teller distortion](https://en.wikipedia.org/wiki/Jahn%E2%80%93Teller_effect), and Q<sub>4</sub> to Q<sub>6</sub> are angular shear modes. To calculate these modes in the way van Vleck proposed in his original paper requires a Cartesian basis with the origin as the centre of the octahedron and the core-ligand bonds along the axes of the basis. Determination of the basis is complicated when there is angular distortion, as there is no way to orient the octahedron in the basis such that all core-ligand bonds are along axes. VanVleckCalculator is a tool to overcome this challenge and calculate the van Vleck distortion modes of an octahedron by rotating the octahedron to minimise the deviation of each ligand from the axes.

In addition to the van Vleck modes, this same rotation algorithm is used to calculate the angular shear, anti-shear, and angular shear fraction defined in the method paper accompanying this code package (under review).

For details on the application and interpretation of the modes, we refer the reader to the method paper accompanying this code package (under review) and the literature. This section will explain the details of how the algorithm works.

##### 2.3.2 Origin fixing

The first step, before performing the octahedral rotation, is to fix the origin. By default, this position is the atom in the centre of the octahedron. However, for some purposes this is not appropriate. For instance, if one is analysing the output of a big box Pair Distribution Function refinement or a Monte Carlo simulation, the central ion may exhibit some thermal motion from its ideal position. This will have a significant effect on the calculated Van Vleck modes, and so in this situation the optional argument should be used to fix the centre of the octahedron to be the crystallographic site. Likewise, the pseudo or second-order Jahn-Teller distortion may factor into the decision about where to situate the origin.

Below are examples of the use of the method `calculate_van_vleck_distortion_modes` for various possible origins. Here the origin is fixed as the position of the central cation (the default):
```
VanVleckModes = oct1.calculate_van_vleck_distortion_modes()
```

where `VanVleckModes` will by a Python list containing [Q<sub>1</sub>,Q<sub>2</sub>,Q<sub>3</sub>,Q<sub>4</sub>,Q<sub>5</sub>,Q<sub>6</sub>].

If you wish to fix the coordinates which are to be the origin, this is done as follows:

```
VanVleckModes = octahedron.calculate_van_vleck_distortion_modes(octahedral_centre=[x,y,x])
```
where `x`, `y`, and `z` are the absolute coordinates of the centre of the octahedron.

Additionally, it is also possible to fix the origin as the average position of the 6 ligands, by fixing the `octahedral_centre` argument as "average_ligand_position". An example of this:

```
VanVleckModes = octahedron.calculate_van_vleck_distortion_modes(octahedral_centre="average_ligand_position")
```

##### 2.3.3 Setting the axes

The mathematics of the Van Vleck modes assumes a perfect octahedron with no angular distortion, i.e. where all ligand-core-ligand angles are an integer number of 90°. In practice, coordination octahedra in crystal systems typically exhibit some angular distortion. For the sake of calculating the van Vleck modes, this gives rise to two possible approaches:
1. Ignore the angular distortion and perform the calculation as if the vector from the centre of the octahedron to each ligand is the axis.
2. Attempt to rotate the octahedron such that each core-ligand bond has an assigned axis, and the average angle between each core-ligand bond and the assigned axis is as low as possible. Then perform the calculation using the coordinate of each ligand in the axis to which the ligand is assigned.

VanVleckCalculator defaults to taking the second approach, using an algorithm which automatically rotates the octahedron to attempt to minimise the angle between each core-ligand bond and the crystallographic axes. However, there is the functionality to take both approaches.

###### 2.3.3.1 Set axes as bond directions

This approach ignores the angular distortion entirely. It can be performed by supplying the argument `ignore_angular_distortion` to the method, for instance:

```
VanVleckModes = oct1.calculate_van_vleck_distortion_modes(ignore_angular_distortion=True)
```

This `ignore_angular_distortion` argument is compatible with the `octahedral_centre` argument for fixing the origin. It is also compatible with the `specified_axes` argument described in the next section. It is not compatible with the `output_pairs` argument.

Note that this approach will always give values of zero for the Q<sub>4</sub>, Q<sub>5</sub>, and Q<sub>6</sub> modes.

###### 2.3.3.2 Perform calculation along orthogonal axes

In order to perform the Van Vleck calculation along orthogonal axes, the octahedron must be rotated such that the three axes of the octahedron correspond to the axes in space. In practice, this is not possible to achieve exactly if an octahedron has angular distortion, and so a match must be made as closely as possible. In VanVleckCalculator, there are three approaches to this problem:
1. By default, the following approach is taken. In each of the _xy_, _xz_, and _yz_ planes, the angle to rotate 4 (the 4 which should end up being within that plane) of the 6 atoms such that within that plane, all atoms are on an axis, is calculated, and the octahedron is then rotated by the average of the four angles. The result is a set of axes optimised for the shape of the octahedron. This approach may not be appropriate for analysis of a supercell from big-box PDF analysis or Monte-Carlo methods, because the random motion of the ligands may lead to some random rotation of the calculated axes of the octahedron relative to the crystallographic axes. If the algorithm fails, a warning will be displayed to the user.
2. The second approach is achieved by manually inputting a set of vectors to the `calculate_van_vleck_distortion_modes()` method using the `specified_axes` argument, and fixing the argument `automatic_rotation=False`. By this approach, three orthogonal vectors are given, and CrystalPolyhedra will then rotate the octahedron such that these three vectors correspond to the _x_-, _y_-, and _z_-axes respectively.
3. The third approach is a combination of the first two. A set of axis are input using the `specified_axes` argument, as in option 2, but the argument `automatic_rotation=True`. The code will rotate the octahedron such that these correspond to the axes in space. However, then the first option is used, but with a starting point closer to the actual axes.

Here the first method is used, i.e. the axes are automatically determined:
```
VanVleckModes = octahedron.calculate_van_vleck_distortion_modes()
```

Here, the second approach is taken:

```
VanVleckModes = octahedron.calculate_van_vleck_distortion_modes(
    specified_axes=[ [1,1,0], [1,-1,0], [0,0,1] ],
    automatic_rotation=False
)
```
where `[ [1,1,0], [1,-1,0], [0,0,1] ]` are presumed to be the axes of the octahedron.

Finally, option 3 is as follows:

```
best_guess_axes = [[1,-np.sqrt(2),1],[1,np.sqrt(2),1],[-1,0,1]]
VanVleckModes = octahedron.calculate_van_vleck_distortion_modes(specified_axes=best_guess_axes)
```

To check that this operation was successful, three methods can be used to check the position of ligands around the centre following the axis rotation. These three methods take all the same arguments as the methods which calculate the van Vleck modes.
- `visualise_sites_for_van_vleck()` method can be used to produce a 3D plot showing the positions of the pairs (each shown in a different colour) around the centre of mass; to check the rotation worked, make sure that the ligand positions are as close as possible to the axes.
- `output_sites_for_van_vleck()` method can be used to return a Python list with shape (3,2,3) where the first axis contains three pairs of opposite ligands, the second axis refers to each site in a pair, and the third axis is the _x_-, _y_-, and _z_- positions of the site.
- `output_and_visualise_sites_for_van_vleck()` method performs both the above operations.

An alternative approach to checking the pair positions would be to use the `output_pairs` argument, which gives a second output identical to the `output_sites_for_van_vleck()` method. For example:

```
VanVleckModes, pair_positions = oct1.calculate_van_vleck_distortion_modes(output_pairs=True)
```

Finally, a rotational tolerance parameter can be set, which determines whether the rotation is judged to have been successful. The default value is 0.25. This can be set with the argument `rotation_tolerance` given to any of the van Vleck methods described in this section. The boolean argument `omit_failed_rotations` can be set to True if any calculations where the rotation is deemed to have failed (as defined by the `rotation_tolerance`) will return None for all parameters. This may be useful when analysing many octahedra within a supercell.

##### 2.3.4 Methods which use this rotation algorithm

###### 2.3.4.1 calculate_van_vleck_distortion_modes

The `calculate_van_vleck_distortion_modes()` method calculates the 6 modes, and can be implemented as follows:

```
modes = oct1.calculate_van_vleck_distortion_modes()
```

###### 2.3.4.2 calculate_van_vleck_jahn_teller_params

The `calculate_van_vleck_jahn_teller_params()` method returns two parameters based on the magnitude √(Q<sub>2</sub> + Q<sub>3</sub>) and the angle arctan(Q<sub>2</sub>/Q<sub>3</sub>) of the Jahn-Teller modes Q<sub>2</sub> and Q<sub>3</sub>. This angle will be in the range 0 to 2π. This will return the angle in radians, but using an optional argument `degrees=True` will give an angle in radians. I.e.:

```
params = oct1.calculate_van_vleck_jahn_teller_params()
```

will give [mag,angle] where angle is in radians. Degrees can be obtained as follows:

```
params = oct1.calculate_van_vleck_jahn_teller_params(degrees=True)
```

###### 2.3.4.3 calculate_degenerate_Q3_van_vleck_modes

Another Van Vleck calculation method is `calculate_degenerate_Q3_van_vleck_modes()` which returns a list containing [Q<sub>3</sub>,-0.5Q<sub>3</sub>+0.5√3 Q<sub>2</sub>,-0.5Q<sub>3</sub>-0.5√3 Q<sub>2</sub>], which are degenerate and equivalent modes for the three possible axes of elongation/compression.

###### 2.3.4.4 calculate_angular_shear and calculate_angular_antishear

###### 2.3.4.5 calculate_angular_shear and calculate_angular_antishear

##### 2.3.5 Reference

The van Vleck modes were introduced in [_Van Vleck, J. H. "The Jahn‐Teller Effect and Crystalline Stark Splitting for Clusters of the Form XY6." The Journal of Chemical Physics 7.1 (1939): 72-84._](https://aip.scitation.org/doi/abs/10.1063/1.1750327) The first known use of the approximation where angular distortion is ignored was in [_Kanamori, Junjiro. "Crystal distortion in magnetic compounds." Journal of Applied Physics 31.5 (1960): S14-S23._](https://pubs.aip.org/aip/jap/article-abstract/31/5/S14/147388/Crystal-Distortion-in-Magnetic-Compounds)

#### 2.4 Calculating Octahedral parameters

Once an `Octahedron` object is initialised, various parameters can be calculated.

#### 2.4.1 Octahedral volume

The volume of a tetrahedron is calculated from the distances between its vertices using the [Cayley-Menger Determinant](https://mathworld.wolfram.com/Cayley-MengerDeterminant.html). For an `Octahedron` object, an octahedron is split into 8 tetrahedra and the volume of each of these is calculated. Any other polyhedron has its volume determined by splitting the surface into triangles, and calculating the volume of the tetrahedron bound by the vertices of the triangle and the centre of the polyhedron.

To obtain the volume, the `calculate_volume()` method should be used. This can be done as follows:
```
oct1.calculate_volume()
```

Note volume is given in Ångstroms cubed.

This approach to the calculation was inspired by the work in [Swanson, Donald K., and R. C. Peterson. "Polyhedral volume calculations." The Canadian Mineralogist 18.2 (1980): 153-156.](https://pubs.geoscienceworld.org/canmin/article/18/2/153/11407/Polyhedral-volume-calculations)

#### 2.4.2 External surface area

Octahedral surface area is defined by splitting up the surface area into a set of triangles and summing the area of these triangles. It can be done using the `calculate_surface_area()` method. This can be done as follows:

```
oct1.calculate_surface_area()
```

where the returned value is in units of Ångstroms squared.

This approach to the calculation was inspired by the work in [Swanson, Donald K., and R. C. Peterson. "Polyhedral volume calculations." The Canadian Mineralogist 18.2 (1980): 153-156.](https://pubs.geoscienceworld.org/canmin/article/18/2/153/11407/Polyhedral-volume-calculations)

#### 2.4.3 Bond length distortion index

The bond length distortion index parametrises the distortion in bond length from the centre of the octahedron. It takes the average of the differences between each bond length (from centre to vertex) and the average bond length. It is defined as follows:

$$D=\frac{1}{n}\sum_i^n \frac{|l_i - l_\mathrm{av}|}{l_\mathrm{av}} $$

where $n$ is the number of ligands around the central atom; for an octahedron, $n=6$.

It can be calculated using the `calculate_bond_length_distortion_index()` parameter as follows:

```
oct1.calculate_bond_length_distortion_index()
```

Another variation involves calculating this same parameter but using the average position of the 6 ligands as an alternative to using the central atom. This may be useful for instance for the second-order Jahn-Teller distortion, when the central atom is offset. This can be achieved using the `octahedral_centre` string argument as follows:

```
polyhedron.calculate_bond_length_distortion_index(octahedral_centre = "average_ligand_position")
```

It was first defined in [_Baur, W. H. "The geometry of polyhedral distortions. Predictive relationships for the phosphate group." Acta Crystallographica Section B: Structural Crystallography and Crystal Chemistry 30.5 (1974): 1195-1215._](https://scripts.iucr.org/cgi-bin/paper?a11025)

#### 2.4.4 Bond angle variance

Bond angle variance parameterises the degree of angular distortion from a perfect octahedron. The vertex-core-vertex angles in a perfect octahedron 90° respectively. In the presence of external constraints, these angles may be distorted. It is defined as follows:

$$\sigma^2 = \frac{1}{m-1} \sum_{i=1}^m (\phi_i - \phi_0)^2$$

where $m$ is the number of bond angles (i.e. 12 for octahedra), $\phi_i$ is bond angle $i$, and $\phi_0$ is the ideal bond angle for a regular octahedron (i.e. 90°).

It can be calculated using the `calculate_bond_angle_variance()` parameter as follows:

```
oct1.calculate_bond_angle_variance()
```

Output can be calculated in radians by using the optional argument `degrees` set to False, i.e. `calculate_bond_angle_variance(degrees=False)`.

It was first defined in [_Baur, W. H. "The geometry of polyhedral distortions. Predictive relationships for the phosphate group." Acta Crystallographica Section B: Structural Crystallography and Crystal Chemistry 30.5 (1974): 1195-1215._](https://scripts.iucr.org/cgi-bin/paper?a11025)

#### 2.4.5 Quadratic elongation

Quadratic elongation parameterises the elongation of a polyhedron. It is defined as follows:

$$<\lambda > = \frac{1}{n} \sum_{i=1}^n \left(\frac{l_i}{l_0}\right)^2$$

where $l_0$ is the centre-to-vertex distance of a regular polyhedron of the same volume.

It can be calculated using `calculate_quadratic_elongation()` as follows:

```
oct1.calculate_quadratic_elongation()
```

It was first defined in [_Robinson, Keith, G. V. Gibbs, and P. H. Ribbe. "Quadratic elongation: a quantitative measure of distortion in coordination polyhedra." Science 172.3983 (1971): 567-570_](https://www.science.org/doi/abs/10.1126/science.172.3983.567)

#### 2.4.6 Effective coordination number

Effective coordination number parameterises the bond length distortion of a polyhedron in an equivalent manner to the bond length distortion index. It is defined as follows:

$$\mathrm{ECoN} = \sum_{i=1}^n \exp \left[ 1 - \left(\frac{l_i}{l'_\mathrm{av}} \right)^6 \right]$$

where, $l'_\mathrm{av}$ is not a normal mean average, but a weighted average bond length:

$$l'_\mathrm{av} =
\frac{  \sum_{i=1}^n l_i \exp \left[ 1 - \left(\frac{l_i}{l_\mathrm{min}} \right)^6  \right]}
{ \sum_{i=1}^n   \exp \left[ 1 - \left(\frac{l_i}{l_\mathrm{min}} \right)^6 \right]}$$

It can be calculated using `calculate_effective_coordination_number()` as follows:

```
oct1.calculate_effective_coordination_number()
```

It was first defined by Rudolf Hoppe in a 1979 work, see later paper here: [Hoppe, Rudolf, et al. "A new route to charge distributions in ionic solids." Journal of the Less Common Metals 156.1-2 (1989): 105-122.](https://www.sciencedirect.com/science/article/pii/0022508889904116)

##### 2.4.7 Off-centering metrics

For materials exhibiting a second-order Jahn-Teller distortion, it is useful to parameterise the degree of this distortion using the off-centering metric. Two examples of this are given in the literature.

###### 2.4.7.1 Off-centering distance

An off-centering distance has been defined in the literature. This is given by the following equation:

$$d_{oct} = \left| R_\mathrm{centre} - \frac{1}{6}\sum R_\mathrm{ligand}  \right| $$

It can be calculated in VanVleckCalculator as follows:

```
off_centre_dist = oct1.calculate_off_centering_distance()
```

This is described in [_Koçer, Can P., et al. "Cation disorder and lithium insertion mechanism of Wadsley–Roth crystallographic shear phases from first principles." Journal of the American Chemical Society 141.38 (2019): 15121-15134._](https://pubs.acs.org/doi/full/10.1021/jacs.9b06316)

###### 2.4.7.2 Off-centering metric

An alternative metric was defined by PS Halasyamani in 2004. First, three angles $\theta_1$, $\theta_2$, and $\theta_3$ are defined; these are the angles between opposite ligands in an octahedron via the core site, i.e. these angles which would be 180° for a perfect, undistorted, regular octahedron. For each $i$th angle, two distances $R_i^A$ and $R_i^B$ are defined which is the distances between the core site and each of the two ligands giving rise to a particular value of $\theta$.

Given these values, the off-centering metric can then be calculated as follows:

$$\Delta_d = |\frac{R_1^A - R_1^B}{\cos(\theta_1)}|    +      |\frac{R_2^A - R_2^B}{\cos(\theta_2)}|     +     |\frac{R_3^A - R_3^B}{\cos(\theta_3)}| $$

It can be calculated in VanVleckCalculator as follows:

```
off_centre_param = oct1.calculate_off_centering_metric()
```

This is described in [_Halasyamani, P. Shiv. "Asymmetric cation coordination in oxide materials: Influence of lone-pair cations on the intra-octahedral distortion in d<sub>0</sub> transition metals." Chemistry of Materials 16.19 (2004): 3586-3592._](https://pubs.acs.org/doi/abs/10.1021/cm049297g)

## 3 Citation

If you use this code in your work, please cite both this GitHub repository and the paper _L. A. V. Nagle-Cocco and S. E. Dutton, Under Review (2023)._
