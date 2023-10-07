# -*- coding: utf-8 -*-

"""
VanVleckCalculator is a Python code which defines an Octahedron class, 
from which various parameters can be calculated, in particular the van
Vleck distortion modes.

This code was written by Liam Nagle-Cocco at the University of 
Cambridge in 2023.
"""

#import main packages
import pymatgen as pymatgen
import numpy as np
import warnings
import copy

GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD = 0.25
GLOBAL_PARAM__ROTATION_TOLERANCE_AXES = 3e-4

class Octahedron(object):

  # Ideal angle is the ideal ligand-core_site-ligand angle, in degrees
  ideal_angle = 90

  # number_of_smallest_angles is the number of ligand-core_site-ligand 
  #   angles for neighbouring ligands
  number_of_smallest_angles = 12

  # number of ligands
  number_of_ligands = 6

  def __init__(
    self,
    core_site,
    structure,
    ligands_min_distance=0.0,
    ligands_max_distance=2.7,
    forbidden_ligands=[],
    possible_ligands=None
  ):
    """
    The initialisation of an instance of the Octahedron class.
    
    Arguments:
    -core_site: the atom in the centre (pymatgen PeriodicSite object)
    -structure: a pymatgen Structure object corresponding to the unit  
      cell containing the core_site
    -ligands_min_distance: an optional argument (defaults to 0.0 if not 
      includes) which gives the minimum distance from core_site (in 
      Angstroms) for atoms to be considered possible ligands.
    -ligands_max_distance: an optional argument (defaults to 2.7 if not
      included) which gives the maximum distance from core_site (in 
      Angstroms) for atoms to be considered possible ligands.
    -forbidden_ligands: an optional argument which lists the names of 
      species which specifically cannot be considered as ligands; for 
      instance, if the core_site is "Ni" you may which to exclude "Ni" 
      from being considered as a possible ligand. If not included, 
      defaults to an empty list.
    -possible_ligands: an optional argument which lists atom types 
      which can be considered Ligands. Defaults to a None value in 
      which case all species which are not listed in forbidden_ligands
      are considered as possible ligands.
    """

    self.__check_inputs(
      core_site,
      structure,
      ligands_min_distance,
      ligands_max_distance,
      forbidden_ligands,possible_ligands
    )

    self.core_site = core_site
    self.struct = structure
    self.ligands = self.get_neighbours(
      ligands_min_distance,
      ligands_max_distance,
      forbidden_ligands,
      possible_ligands
    )

    self.core_ligand_bond_lengths = self.__return_bond_lengths()
    self.volume = self.calculate_volume()
    if np.isnan(self.volume):
      warnings.warn(
        "Warning: volume calculation failed. Volume is not a number."
      )

  def __check_inputs(
    self,
    core_site,
    struct,
    ligands_min_distance,
    ligands_max_distance,
    forbidden_ligands,
    possible_ligands,
    ):
    """
    This method checks the inputs given to the __init__ to make sure they 
      are valid for generating an octahedron.
    
    Arguments:
    -core_site: the atom in the centre (pymatgen PeriodicSite object)
    -structure: a pymatgen Structure object corresponding to the unit  
      cell containing the core_site
    -ligands_min_distance: an optional argument (defaults to 0.0 if not 
      includes) which gives the minimum distance from core_site (in 
      Angstroms) for atoms to be considered possible ligands.
    -ligands_max_distance: an optional argument (defaults to 2.7 if not
      included) which gives the maximum distance from core_site (in 
      Angstroms) for atoms to be considered possible ligands.
    -forbidden_ligands: an optional argument which lists the names of 
      species which specifically cannot be considered as ligands; for 
      instance, if the core_site is "Ni" you may which to exclude "Ni" 
      from being considered as a possible ligand. If not included, 
      defaults to an empty list.
    -possible_ligands: an optional argument which lists atom types 
      which can be considered Ligands. Defaults to a None value in 
      which case all species which are not listed in forbidden_ligands
      are considered as possible ligands.
    
    Exception:
      This method will throw an exception if any of the arguments are 
        not in the correct format.
    """

    if (
      (type(ligands_min_distance) not in [type(1),type(1.0)]) 
      or (ligands_min_distance < 0)
    ):
      msg =  "Ligand min distance is not valid."
      msg += "\nIt must be a positive number, denoting maximum distance in Angstroms."
      raise ValueError(msg)

    if ligands_min_distance > ligands_max_distance:
      msg = "Maximum possible core-ligand distance cannot be less" 
      msg += "than the minimum possible core-ligand distance."
      raise ValueError(msg)

    if (
      (type(ligands_max_distance) not in [type(1),type(1.0)]) 
      or (ligands_max_distance < 0)
    ):
      msg = "Ligand max distance is not valid.\nIt must be a positive number"
      msg += ", denoting maximum distance in Angstroms."
      raise ValueError(msg)

    if (
      (type(forbidden_ligands) != type([]))
      or (not all([type("") == type(i) for i in forbidden_ligands]))
    ):
      msg = "forbidden_ligands must be either an empty list, or a list of strings"
      msg += ", with each string representing a species forbidden from being a ligand."
      raise ValueError(msg)

    if (
      (type(possible_ligands) not in [type([]),type(None)])
      or (
        (type(possible_ligands) == type([])) 
        and (not all([type("x") == type(i) for i in possible_ligands]))
      )
    ):
      msg = "The parameter possible_ligands must be either None, to mean any species may be a ligand"
      msg += ", or a list of strings where each string is a species which may be a ligand."
      raise ValueError(msg)
    
    if len(struct) == 0:
      raise ValueError("Error: structure is empty.")

    if type(core_site)!=type(struct[0]):
      raise ValueError("Error: core_site must be PeriodicSite.")

    if core_site not in struct:
      raise Exception(
        "Core atom must be within unit cell.\nCore atom is at {}.".format(
          core_site
        )
      )
    
  def __find_projection_of_a_vector(self,u,v):
    """
    This helper method calculates the vector u projected along v.

    Arguments:
      u: a vector, in the form of a Python list or numpy array 
        with shape (3,3)
      v: a vector, in the form of a Python list or numpy array 
        with shape (3,3)

    Returns: a vector (i.e. a np.array with three elements)
    """
    #make sure a and b are numpy arrays
    u = np.array(u)
    v = np.array(v)

    #projection
    v_norm = np.sqrt(sum(v**2))    
    proj_of_u_on_v = (np.dot(u, v)/v_norm**2)*v

    return proj_of_u_on_v

  def __find_tetrahedral_volume_edge_lengths(self,u, v, w, U, V, W) :
    """
    A helper method for calculating the volume of a tetrahedron
      given its edge lengths.

    Arguments:
      u,v,w,U,V,W: floats, each representing the edge length of a 
        tetrahedron.

    Return:
      float: the volume of a tetrahedron with edge lengths u, 
        v, w, U, V, W
    """
    uPow = u**2
    vPow = v**2
    wPow = w**2
    UPow = U**2
    VPow = V**2
    WPow = W**2

    a = (
      4 * (uPow * vPow * wPow)
      - uPow *(vPow + wPow - UPow)**2
      - vPow * (wPow + uPow - VPow)**2
      - wPow * (uPow + vPow - WPow)** 2
      + (
        (vPow + wPow - UPow) 
        * (wPow + uPow - VPow) 
        * (uPow + vPow - WPow)
      )
    )

    volume = (a**0.5) / 12

    return volume

  def __find_tetrahedral_volume(self,atom1,atom2,atom3,atom4):
    """
    This calculates the volume of a tetrahedral volume with 4 atoms as 
      vertices. 
    
    Arguments:
      4 PeriodicSite objects
      
    Return:
      float: the volume of a tetrahedron bound by 4 atoms.
    """
    u = self.get_intersite_distance(atom1,atom3)
    v = self.get_intersite_distance(atom1,atom2)
    w = self.get_intersite_distance(atom1,atom4)
    U = self.get_intersite_distance(atom2,atom3)
    V = self.get_intersite_distance(atom2,atom4)
    W = self.get_intersite_distance(atom4,atom3)

    volume = self.__find_tetrahedral_volume_edge_lengths(
      u, v, w, U, V, W
    )

    return volume

  def calculate_core_ligand_distance_for_perfect_octahedron(self):
    """
    This method will calculate the core-ligand distance for a perfect
      octahedron of equal volume to the octahedron.
    
    A perfect octahedron can be split into two square-based pyramids 
      with base area a.
      Hence the volume is V=sqrt(2)*a^3/3
      from Pythagoras's theorem, 2a^2 = (2l_0)^2 so a = sqrt(2)*l_0
      therefore l_0 = (6*V)^(1/3)

    return:
      float: the core-ligand distance for a perfect octahedron of 
        equal volume to the octahedron, in Angstroms.
    """
    l_0 = (3.0*self.volume/4.0)**(1.0/3.0)
    return l_0

  def calculate_quadratic_elongation(self):
    """
    This method calculates the quadratic elongation of the octahedron, 
      as defined in:
        Robinson, Keith, G. V. Gibbs, and P. H. Ribbe.
        "Quadratic elongation: a quantitative measure of distortion 
        in coordination polyhedra."
        Science 172.3983 (1971): 567-570.
    
    Return:
      float: the quadratic elongation of the octahedron.
    """

    l_0 = self.calculate_core_ligand_distance_for_perfect_octahedron()
    return (
      (1.0/self.number_of_ligands) * ( 1.0 / (l_0*l_0) )
      * np.sum(np.square(self.core_ligand_bond_lengths))
    )
    

  def get_intersite_distance(
    self,
    site1,
    site2,
    override_lattice_check=False
  ):
    """
    This method will give the distance between two sites. Pymatgen 
      does include functionality to do this, but it does not work well 
      when it encounters periodicity, i.e. if a neighbour is in a 
      neighbouring unit cell. This custom method should therefore be 
      used.
    
    Arguments:
      site1: a Pymatgen PeriodicSite object
      site2: a Pymatgen PeriodicSite object
      override_lattice_check: bool. optional, defaults to False. If 
        True, there will be no check to make sure site1 and site2 are 
        in the same cell.

    Return: 
      float: the distance (in Angstroms) between the two sites.
    """

    if not override_lattice_check:
      if site1.lattice != site2.lattice:
        raise ValueError(
          "Error: site1 and site2 are not in the same cell."
        )
    
    return np.sqrt(
      (
        site1.coords[0]-site2.coords[0])**2
        + (site1.coords[1]-site2.coords[1])**2
        + (site1.coords[2]-site2.coords[2])**2
    )

  def get_neighbours(
      self,
      ligands_min_distance=0.0,
      ligands_max_distance=2.7,
      forbidden_ligands=[],
      possible_ligands=None
    ):
    """
    This method will list all ligands, in ascending order of bond 
      length from the core_site, provided they meet the criteria 
      given in the __init__ method.

    Arguments (all optional):
      ligands_min_distance (float): the minimum distance of a site to 
        be considered a neighbour (in Angstroms)
      ligands_max_distance (float) the maximum distance of a site to 
        be considered a neighbour (in Angstroms)
      forbidden_ligands (list): a list of strings or site species which 
        may not be considered neighbours
      possible_ligands (list or None): a list of strings or site 
        species which only may be considered neighbours. Defaults to 
        None, meaning no restriction
      
    Return:
      Python list: containing PeriodicSite objects
    """
    ligands = []
    all_neighbors = self.struct.get_neighbors(
      self.core_site, 
      ligands_max_distance
    )
    for i in all_neighbors:

      species_name = str(i.species)
      for j in ['.','+','-','0','1','2','3','4','5','6','7','8','9']:
        species_name = species_name.replace(j, '')

      # check that this atom is not forbidden from being a ligand
      if (str(species_name) not in forbidden_ligands): 
         
         #check atom is in possible_ligands
         if (
          (possible_ligands==None) 
          or (str(species_name) in possible_ligands)
        ): 

          #check ligand is further than the minimum distance
          distance = self.get_intersite_distance(i,self.core_site)
          if distance > ligands_min_distance: 
            ligands.append(i)

    ligands.sort(
      key = lambda x : self.get_intersite_distance(x,self.core_site)
    )
    cropped_ligands = [
      ligands[i] for i in range(len(ligands)) if i<self.number_of_ligands
    ]

    if len(cropped_ligands) != self.number_of_ligands:
      err = "There are not the correct number of ligands for atom site {}.\nThere should be {}. There are {}.".format(
        str(self.core_site),
        str(self.number_of_ligands),
        str(len(cropped_ligands))
      )
      raise Exception(err)

    return cropped_ligands

  def __return_bond_lengths(self):
    """
    This method returns a numpy array of floats representing bond 
      lengths between the core_site and the ligands.

    return:
      list of floats: in ascending order, bond lengths between 
        core_site and ligands, in Angstroms
    """
    bonds = []
    for i in self.ligands:
      bond = self.get_intersite_distance(self.core_site,i)
      bonds.append(bond)
    return np.array(bonds)

  def calculate_average_ligand_bond_length(
      self,
      octahedral_centre="core_atom",
    ):
    """
    Calculates the average bond length between the centre of the 
      octahedron and its ligands.

    Arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.

    Returns:
      float: The average bond length.
    """
    #check inputs
    if type(octahedral_centre) not in [type("xyz"),type([]),type(np.array([]))]: 
      raise ValueError("Error: octahedral_centre must be string or list.")
    
    if type(octahedral_centre) in [type([]),type(np.array([]))]:

      if len(octahedral_centre)!=3:
        raise ValueError(
          "Error: octahedral_centre must be of the form [x,y,z]."
        )
      
      if not all([
        type(octahedral_centre[i]) 
        in [type(0.5),type(1),np.float64] for i in [0,1,2]
      ]):
        
        err = "Error: octahedral_centre must contain three floats or integers but contains {}."
        raise ValueError(
          err.format( [type(i) for i in octahedral_centre] )
        )
    else:
      if octahedral_centre not in ["core_atom","average_ligand_position"]:
        raise ValueError(
          "Error: octahedral_centre must by either core_atom or average_ligand_position."
        )
    
    #implementation
    if octahedral_centre=="average_ligand_position":
      average_ligand_position = [
        np.sum(
          [j.coords[i] for j in self.ligands]
        )/len(self.ligands) for i in [0,1,2]
      ]
      lengths = [
        np.sqrt(
          (
            i.coords[0]-average_ligand_position[0])**2 
            + (i.coords[1]-average_ligand_position[1])**2 
            + (i.coords[2]-average_ligand_position[2])**2
          ) for i in self.ligands
      ]
    elif type(octahedral_centre)!=type("xyz"):
      lengths = [
        np.sqrt(
          (
            i.coords[0]-octahedral_centre[0])**2 
            + (i.coords[1]-octahedral_centre[1])**2 
            + (i.coords[2]-octahedral_centre[2])**2
          ) for i in self.ligands
      ]
    else:
      lengths = self.core_ligand_bond_lengths
      
    return np.average(lengths)

  def calculate_bond_length_distortion_index(
      self,
      octahedral_centre="core_atom",
  ):
    """
    Calculates and returns the bond length distortion index of a 
      polyhedron as defined in Baur (1974).
    The bond length distortion index is calculated as the average 
      absolute deviation of individual centre-ligand bond lengths 
      from the average bond length, normalized by the average
      bond length.

    Reference:
      Baur, W. H. "The geometry of polyhedral distortions. Predictive 
      relationships for the phosphate group." Acta Crystallographica 
      Section B: Structural Crystallography and Crystal Chemistry,
      1974, 30(5), 1195-1215.
    
    Arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.

    Returns:
      float: the bond length distortion index.
    """

    #check inputs
    if type(octahedral_centre) not in [type("xyz"),type([]),type(np.array([]))]: 
      raise ValueError("Error: octahedral_centre must be string or list.")
    
    if type(octahedral_centre) in [type([]),type(np.array([]))]:

      if len(octahedral_centre)!=3:
        raise ValueError(
          "Error: octahedral_centre must be of the form [x,y,z]."
        )
      
      if not all([
        type(octahedral_centre[i]) 
        in [type(0.5),type(1),np.float64] for i in [0,1,2]
      ]):
        
        err = "Error: octahedral_centre must contain three floats or integers but contains {}."
        raise ValueError(
          err.format( [type(i) for i in octahedral_centre] )
        )
    else:
      if octahedral_centre not in ["core_atom","average_ligand_position"]:
        raise ValueError(
          "Error: octahedral_centre must by either core_atom or average_ligand_position."
        )

    if octahedral_centre=="average_ligand_position":
      average_ligand_position = [
        np.sum(
          [j.coords[i] for j in self.ligands]
        )/len(self.ligands) for i in [0,1,2]
      ]
      lengths = np.array([
        np.sqrt(
          (
            i.coords[0]-average_ligand_position[0])**2 
            + (i.coords[1]-average_ligand_position[1])**2 
            + (i.coords[2]-average_ligand_position[2])**2
          ) for i in self.ligands
      ])
    else:
      lengths = self.core_ligand_bond_lengths

    av = np.average(lengths)

    summation = np.sum( abs(lengths-av)/av )

    return (1/len(lengths)) * summation

  def get_bond_angle(self,ligand1,ligand2,degrees=True):
    """
    Calculates the angle between the bonds of two ligand ions 
      to the core atom.

    Arguments:
        ligand1 (PeriodicSite): the first ligand ion
        ligand2 (PeriodicSite): the second ligand ion
        degrees: if True, output is in degrees. Otherwise output 
          is radians. Defaults to True

    Returns:
        float: the angle between the bonds in degrees
    """

    #check inputs
    if not type(degrees)==type(True):
      raise ValueError("Error: degrees must be boolean")
    for i in [0,1]:
      ligand = [ligand1,ligand2][i]
      if type(ligand) not in [
        pymatgen.core.structure.Neighbor,
        pymatgen.core.structure.PeriodicNeighbor,
        type(self.core_site)
      ]:
        type_ = type(ligand)
        raise ValueError(
          "Error: ligand{} not Neighbor or PeriodicSite. Type is {}".format(
            str(i+1),
            type_
          )
        )

    l1_coords = (
      (ligand1.coords - self.core_site.coords) 
      / np.linalg.norm(ligand1.coords - self.core_site.coords)
    )
    l2_coords = (
      (ligand2.coords - self.core_site.coords) 
      / np.linalg.norm(ligand2.coords - self.core_site.coords)
    )

    dot_product = np.dot(l1_coords, l2_coords)

    # fix to 1 to prevent floating point error
    if abs(dot_product) > 1: 
      dot_product /= abs(dot_product)
    angle = np.arccos(dot_product)

    if degrees:
      angle = angle / np.pi * 180.0

    return(angle)

  def list_all_ligand_core_ligand_angles(self,degrees=True):
    """
    Returns an array containing all bond angles (including bond angles 
      between opposite ligands).

    Arguments:
      degrees (bool): an optional argument, defaults to True. 
        Calculated values are degrees (True) or radians (False)
    """
    if type(degrees)!=type(True):
      raise ValueError(
        "Error: degrees argument must be True or False."
      )

    angles = []
    for i in range(self.number_of_ligands-1):
      for j in range(i+1,self.number_of_ligands):
        if (i != j):
          angle = self.get_bond_angle(
            self.ligands[i],
            self.ligands[j],
            degrees=degrees
          )
          angles.append(angle)
    return(angles)

  def calculate_bond_angle_variance(self,degrees=True):
    """
    Calculates the bond angle variance, as defined in
        Baur, W. H. "The geometry of polyhedral distortions. 
        Predictive relationships for the phosphate group."
        Acta Crystallographica Section B: Structural Crystallography 
        and Crystal Chemistry 30.5 (1974): 1195-1215.
    
    Arguments:
      degrees (bool): an optional argument, defaults to True. 
        Calculated value is degrees (True) or radians (False)
    
    return:
      float: the bond angle variance, in units of either degrees or
        radians depending on user input
    """
    if type(degrees)!=type(True):
      raise ValueError(
        "Error: degrees argument must be True or False."
      )

    unsorted_angles = self.list_all_ligand_core_ligand_angles(
      degrees=degrees
    )
    angles = sorted(unsorted_angles,reverse=False)
    angles_cropped = angles[:self.number_of_smallest_angles]

    multiplier = (1.0/(float(self.number_of_smallest_angles)-1.0))
    output = multiplier * np.sum(
      [(i-self.ideal_angle)**2 for i in angles_cropped]
    )
    return output

  def calculate_effective_coordination_number(self):
    """
    This method calculates the effective coordination number of the 
      octahedron as first defined in the following paper:
        Hoppe, Rudolf. "Effective coordination numbers (ECoN) and 
        mean fictive ionic radii (MEFIR)." Zeitschrift für 
        Kristallographie-Crystalline Materials 150.1-4 (1979): 23-52.

    return:
      float: the effective coordination number
    """

    min_bond = np.min(self.core_ligand_bond_lengths)
    l_av_exp = np.exp(
      1-np.power(self.core_ligand_bond_lengths/min_bond,6)
    )

    l_av = (
      np.sum(self.core_ligand_bond_lengths * l_av_exp) 
      / np.sum(l_av_exp)
    )

    ECoN = np.sum(
      np.exp(1-np.power(self.core_ligand_bond_lengths/l_av,6))
    )

    return ECoN

  def calculate_off_centering_distance(self):
    """
    This method calculates the off-centering distance defined in Eq 7 
      of:
        Koçer, Can P., et al.
        "Cation disorder and lithium insertion mechanism of 
        Wadsley–Roth crystallographic shear phases from first 
        principles."
        Journal of the American Chemical Society 141.38 
        (2019): 15121-15134.

    return:
      float: the off-centering distance (in Angstroms)
    """
    mean_ligand_position = np.array(
      [
        np.mean([ligand.coords[j] for ligand in self.ligands]) 
        for j in [0,1,2]
      ]
    )

    return np.sqrt(np.sum(
      ((np.array(self.core_site.coords)
        - mean_ligand_position))**2
      )
    )

  def __find_surface_triangles_FindOffset(self,plane,atom):
    """
    This helper method for __find_surface_triangles returns the 
      offset for a given plane, provided the plane contains a given 
      atom

    arguments:
      plane: a Python list containing three floats, which represents 
        the vector perpendicular to the plane
      atom: a pymatgen PeriodicSite or Neighbor object

    return:
      float: a value representing the minimum distance (offset)
        between the origin in space and the plane.
    """
    return np.dot(
      plane,
      np.array(atom.coords)
    )

  def __find_surface_triangles_CrossAndDot(self,atom1,atom2,atom3):
    """
    This helper method for __find_surface_triangles returns the 
      cross_product and offset of a plane containing three atoms

    Arguments:
      atom1: PeriodicSite or Neighbor object
      atom2: PeriodicSite or Neighbor object
      atom3: PeriodicSite or Neighbor object

    return:
      a Python list with three floats, representing a vector 
        perpendicular to a plane
      a float: the minimum distance (offset) between the vector 
        and plane
    """
    cross_product = np.cross(
      np.array(atom1.coords)-np.array(atom2.coords),
      np.array(atom1.coords)-np.array(atom3.coords)
    )
    offset = self.__find_surface_triangles_FindOffset(
      cross_product,
      atom1
    )
    return cross_product, offset

  def __find_surface_triangles_checkOverlap(
      self,
      triangle,
      plane,
      offset,
      triangles,
      planes,
      offsets
    ):
    """
    This helper method for __find_surface_triangles checks to make 
      sure that a new triangle is not coplanar with any pre-existing 
      triangles. If it is, it then makes sure the triangle doesn't 
      overlap.

    This is implemented by checking whether the shared edge of the 
      triangle is the largest edge of the triangle. If it is not, an 
      overlap is assumed.
    
    Arguments:
      triangle: a list containing three PeriodicSite or Neighbor
        objects
      plane: a list of three floats, representing a vector 
        perpendicular to the plane
      offset: a float, the minimum distance between the origin in 
        space and the plane argument
      triangles: a list of triangle objects
      planes: a list of lists of three floats. Each list of three 
        floats represents a vector perpendicular to a plane
      offsets: a list of floats, representing the offsets between 
        each plane in planes and the origin
    
    Return:
      boolean: True if the triangle argument is co-planar 
        with any of the triangles in the triangles argument  
    """

    for l in range(len(triangles)):

      #check if a triangle in the same plane has already been added
      if (
        all(
          [(abs(planes[l][m] - plane[m]) < 0.00001) for m in [0,1,2]]
        )
        and ( abs(offsets[l] - offset) < 0.00001)
      ):

        # identify the two ligands which occur in both triangles
        double_counted = [] 
        single_counted = []

        for m in self.ligands:
          if (m in triangle) and (m in triangles[l]):
            double_counted.append(m)
          if (
            ((m in triangle) and (m not in triangles[l])) 
            or ((m not in triangle) and (m in triangles[l]))
          ):
            single_counted.append(m)

        if not ((len(double_counted)==2) and (len(single_counted)==2)):
          raise Exception("Error: double-counting check failed.")

        # if the shared edge between the two triangles is NOT the 
        #   hypotenuse, then we will consider the triangles overlap
        shared_length = self.get_intersite_distance(
          double_counted[0],
          double_counted[1]
        )
        for m in [0,1]:
          for n in [0,1]:
            if shared_length < self.get_intersite_distance(
              double_counted[m],
              single_counted[n]
            ):
              return True

    return False

  def __find_surface_triangles(self):
    """
    This helper method finds all the surface triangles, following the 
      method described in:
      Swanson, Donald K., and R. C. Peterson. "Polyhedral volume 
      calculations."
      The Canadian Mineralogist 18.2 (1980): 153-156.

    This is necessary for generalised volume calculations and surface 
      area calculations. Note, the original paper included a check for 
      concavity, however this is not possible for octahedra so this 
      part of the algorithm is excluded.

    return: 
      a list of "triangles", where each triangle is a list containing
        three PeriodicSite or Neighbor objects.
    """

    #First assume polyhedron is convex and find all the triangles on 
    #   the surface
    triangles = []
    planes = []
    offsets = []

    for i in range(0,len(self.ligands)-2):
      for j in range(i+1,len(self.ligands)-1):
        for k in range(j+1,len(self.ligands)):
          triangle = [self.ligands[i],self.ligands[j],self.ligands[k]]

          # find equation of plane containing all three points:
          cross_product, offset = self.__find_surface_triangles_CrossAndDot(
            triangle[0],
            triangle[1],
            triangle[2]
          )

          #Determine which side of the plane the core_site is
          offset_core_site = self.__find_surface_triangles_FindOffset(
            cross_product,
            self.core_site
          )
          sign = (offset - offset_core_site)

          if sign != 0: # only use triangle if core_site not in plane
            sign /= abs(offset - offset_core_site) # fix to + or - 1

            useTriangle = True

            # Make sure that all ligands are on the same side of the 
            #   plane as core_site
            for l in self.ligands: 
              offset_ligand = self.__find_surface_triangles_FindOffset(
                cross_product,
                l
              )
              ligand_sign = (offset - offset_ligand)

              # ligand_sign should be != 0, but there can be floating  
              #   point rounding errors so set a tolerance instead
              if abs(ligand_sign) > 0.0000001: 
                ligand_sign /= abs(offset - offset_ligand)
                if ligand_sign != sign:
                  useTriangle = False

            # finally, we need to make sure there is no double-counting
            if self.__find_surface_triangles_checkOverlap(
              triangle,
              cross_product,
              offset,
              triangles,
              planes,
              offsets
            ):
              useTriangle = False

            #if all ligands and core_site on the same side of 
            #   the triangle:
            if useTriangle: 
              triangles.append(triangle)
              planes.append(cross_product)
              offsets.append(offset)

    return triangles

  def calculate_surface_area(self):
    """
    Octahedral surface area is defined by splitting up the surface area
      into a set of triangles and summing the area of these triangles, 
      following the method described in:
        Swanson, Donald K., and R. C. Peterson. "Polyhedral volume 
        calculations."
        The Canadian Mineralogist 18.2 (1980): 153-156.
    
    return:
      float: surface area of the exterior triangles of the Octahedron 
        (in Angstroms squared)
    """
    triangles = self.__find_surface_triangles()

    if len(triangles) == 0:
      raise Exception("Error: Triangle-finding algorithm has failed.")

    area = 0

    for triangle in triangles:

      if len(triangles) != 3:
        raise Exception(
          "Error: Triangle-finding algorithm has failed."
        )

      cross_product = np.cross(
        np.array(triangle[0].coords)-np.array(triangle[1].coords),
        np.array(triangle[0].coords)-np.array(triangle[2].coords)
      )

      area += 0.5 * np.linalg.norm(cross_product)
    return area

  def calculate_volume(self):
    """
    This method returns the volume of the octahedron. It first attempts
      to split the octahedron into 8 tetrahedra. If this fails, it uses
      the more generalised approach given in:
        Swanson, Donald K., and R. C. Peterson. "Polyhedral volume 
        calculations."
        The Canadian Mineralogist 18.2 (1980): 153-156.

    return:
      float: octahedral volume in units of Angstrom cubed
    """
    pairs = self.find_ligand_opposites()
    volume = 0

    # calculate volume for one square-based pyramid
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][0], pairs[1][0], pairs[2][0]
    )
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][0], pairs[1][1], pairs[2][0]
    )
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][0], pairs[1][0], pairs[2][1]
    )
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][0], pairs[1][1], pairs[2][1]
    )

    # add the volume for the other square-based pyramid
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][1], pairs[1][0], pairs[2][0]
    )
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][1], pairs[1][1], pairs[2][0]
    )
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][1], pairs[1][0], pairs[2][1]
    )
    volume += self.__find_tetrahedral_volume(
      self.core_site, pairs[0][1], pairs[1][1], pairs[2][1]
    )

    if type(volume) != type(None):
      return volume
    
    else: # if calculation fails
      triangles = self.__find_surface_triangles()
      if len(triangles) == 0:
        raise Exception("Error: Triangle-finding algorithm has failed.")

      volume = 0
      for triangle in triangles:

        if len(triangles) != 3:
          raise Exception(
            "Error: Triangle-finding algorithm has failed."
          )

        tetra_volume = self.__find_tetrahedral_volume(
          self.core_site, 
          triangle[0], 
          triangle[1], 
          triangle[2]
        )
        volume += tetra_volume

      return volume

  def is_identical(self,octahedron):
    """
    This method checks to see whether a given octahedron object is 
      identical to self.

    argument:
      an instance of the Octahedron class, or a subclass

    return: 
      boolean: True, if the Octahedron object is 
        identical
    """
    if type(self) != type(octahedron):
      return False
    if self.number_of_ligands != octahedron.number_of_ligands:
      return False
    identical = all([
      (self.core_site.coords[k] == octahedron.core_site.coords[k]) 
      for k in [0,1,2]
    ])

    return identical

  def __visualise_pairs(self,local_pair_coords):
    """
    This helper method plots the pairs of opposite ligands in the 
      octahedron. It is indended to help with debugging if the 
      pair-finding doesn't work, in the case of irregular octahedra.

    arguments:
      local_pair_coords: a list containing lists of "pairs", where 
        each "pair" is a list containing two "atoms" where each "atom"
        is another list containing three floats which are the x,y,z 
        coordinates of a site. In total the local_pair_coords should 
        have shape (n,2,3) where n is the number of pairs. n should 
        equal 3 for an octahedron.

    return:
      a matplotlib.axes object, which is a 3D plot of all the pairs in 
        the octahedron.
    """
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")

    colours = ["r","b","k","g","c","y"]

    for j in range(len(local_pair_coords)):
      pair = local_pair_coords[j]
      x = [i[0] for i in pair] 
      y = [i[1] for i in pair] 
      z = [i[2] for i in pair]
      ax.scatter(x, y, z, zdir='z',color=colours[j])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

    return ax

  def visualise_pairs(
      self,
      plotted_vectors=None,
    ):
    """
    This method plots a 3D, interactive figure which shows the 
      positions of the pairs relative to the origin.
    It can also plot axes alongside the atomic sites if plotted_vectors
      is included.
    
    argument:
      plotted_vectors: must be a python list or None. If list, must be 
        a list with shape (3,3), representing three vectors which may be
        supplied and will be plotted alongside the points. Defaults to 
        None.
    """
    #check inputs
    if not type(plotted_vectors) in [
      type(None),
      type([]),
      type(np.array([]))
    ]:
      raise ValueError(
        "Error: plotted_vectors must be None or a list of three vectors."
      )
    
    if type(plotted_vectors) != type(None):
      if len(plotted_vectors) != 3:
        raise ValueError(
          "Error: plotted_vectors must contain 3 vectors not {}".format(
            len(plotted_vectors)
          )
        )
      for i in plotted_vectors:
        if type(i) not in [type([]),type(np.array([]))]:
          raise ValueError(
            "Error: plotted_vectors must contain lists not {}.".format(
              type(i)
            )
          )
        if len(i) != 3: 
          raise ValueError(
            "Error: each vector in plotted_vectors must contain 3 elements not {}.".format(
              len(i)
            )
          )
        for element in i:
          if type(element) not in [type(0.5),type(1),np.float64]:
            raise ValueError(
              "Error: each vector element in plotted_vectors must be int or float, not {}.".format(
                type(element)
              )
            )

    #obtain pairs list
    pairs = self.find_ligand_opposites()

    #plot pairs
    ax = self.__visualise_pairs(
      [
        [
          np.array(i[0].coords)-np.array(self.core_site.coords),
          np.array(i[1].coords)-np.array(self.core_site.coords)
        ] for i in pairs
      ]
    )

    #plot plotted_vectors
    if type(plotted_vectors) != type(None):
      average_length = self.calculate_average_ligand_bond_length()
      for vector in plotted_vectors:
        mag_vector = np.sqrt(np.sum([i*i for i in vector]))
        vector = [i/mag_vector for i in vector]
        ax.plot(
          [- average_length * vector[0], + average_length * vector[0]],
          [- average_length * vector[1], + average_length * vector[1]],
          [- average_length * vector[2], + average_length * vector[2]],
          'r-'
        )

  def calculate_off_centering_metric(self):
    """
    This method calculates the off-centering metric defined in 
      page 3588 of:
        Halasyamani, P. Shiv.
        "Asymmetric cation coordination in oxide materials:
        Influence of lone-pair cations on the intra-octahedral 
        distortion in d0 transition metals."
        Chemistry of Materials 16.19 (2004): 3586-3592.

    return:
      float: the value of the off-centering metric 
    """
    pairs = self.find_ligand_opposites()

    #calculate [theta_1, theta_2, theta_3]
    theta_values = np.array(
      [
        self.get_bond_angle(
          pairs[i][0],
          pairs[i][1],
          degrees=False
        ) for i in [0,1,2]
      ]
    ) 

    terms = np.array([abs(
      (self.get_intersite_distance(self.core_site,pairs[i][0])
      - self.get_intersite_distance(self.core_site,pairs[i][1]))
      /
      np.cos(theta_values[i])
      ) for i in [0,1,2]])

    return np.sum(terms)

  def __find_ligand_opposites__duplicate_check(self,pairs):
    """
    This helper method for self.find_ligand_opposites() checks if 
      there are more or less than 3 pairs. If so, it works out 
      which ligands are duplicates and removes the redundant pair.

    argument:
      a list of shape (n,2) where n is the number of pairs, and each
        pair is a list containing two ligands, where a "ligand" is 
        likely a pymatgen PeriodicNeighbor object
    
    action:
      the code will check to see if there are greater than 3 pairs. If
        so, it will look to see if there are any ligands which appear
        in more than one pair. If there are, it will attempt to 
        determine which pair is not a real pair, and remove it.
    """
    #check not fewer than 3 pairs
    if (len(pairs) < 3): 
      raise Exception(
        "Error: pair finding algorithm failed. {} pairs found, not 3."
      )

    #if there are more than 3 pairs, need to remove bad pairs
    while len(pairs) > 3:
      duplicated_ligand = None
      for i in self.ligands:

        count = 0
        for pair in pairs:
          if i in pair:
            count += 1

        if count > 1:
          duplicated_ligand = i
          break

      err = "Error: No ligands are duplicated."
      err += " Polyhedron probably is not an octahedron."
      assert type(duplicated_ligand) != type(None), err

      #all ligand list
      all_ligands = []
      for pair in pairs:
        all_ligands.append(str(pair[0].coords))
        all_ligands.append(str(pair[1].coords))

      #calculate properties of the duplicated pairs
      duplicating_pairs = [
        [
          i,
          pairs[i],
          self.get_intersite_distance(pairs[i][0],pairs[i][1]),
          self.get_bond_angle(pairs[i][0],pairs[i][1]),
          [all_ligands.count(str(j.coords)) for j in pairs[i]]
        ] for i in range(0,len(pairs)) 
        if str(duplicated_ligand.coords) in
        [str(j.coords) for j in pairs[i]]
      ]

      #remove on the basis of ligand count
      if [
        i[4].count(1) == 0 for i in duplicating_pairs
      ].count(True)==1:
        
        bad_index = None
        for i in duplicating_pairs:
          if i[4].count(1) == 0:
            bad_index = i[0]

      else:
        #check which duplicated pair to remove, based on angle and length
        index_max_angle = sorted(
          duplicating_pairs,
          key = lambda x : x[3]
        )[0][0]
        index_max_dist  = sorted(
          duplicating_pairs,
          key = lambda x : x[2]
        )[0][0]
        
        #error message
        if index_max_angle!=index_max_dist:
          err_string = "Error: code can only solve pair duplication if index_max_angle {}==index_max_dist {}"
          err_string = err_string.format(
            index_max_angle,
            index_max_dist
          )

          print(err_string)
          print("There are {} pairs. These are:".format(len(pairs)))
          for i in pairs:
            print(i)
          print("\nThere are {} duplicating_pairs. These are:".format(len(duplicating_pairs)))
          for i in duplicating_pairs:
            print(i)
          assert index_max_angle==index_max_dist, err_string
        
        #save index
        bad_index = index_max_angle

      #remove the unnecessary duplicated pair
      pairs.pop(bad_index)

  def __find_ligand_opposites__find_all_pairs(self):
    """
    This helper method for find_ligand_opposites obtains all pairs
      of opposite ligands for the octahedron.
    
    It may return more or less than three pairs, so subsequent methods
      must check the output is valid.
    """
    pairs = []
    for i in self.ligands:
      local_bond_angles = []
      for j in self.ligands:
        if j != i:
          local_bond_angles.append([j,180-self.get_bond_angle(i,j)])
      local_bond_angles = sorted(
        local_bond_angles,
        key=lambda x: x[1]
      )[:-1]

      # This line sorts the pair so that we don't accidentally get the 
      #   same pair twice with a different order
      # note we sort again later
      unsorted_pair = [i,local_bond_angles[0][0]]
      pair = sorted(
        unsorted_pair, 
        key=lambda x: (x.coords[2],x.coords[1],x.coords[0])
      )

      if pair not in pairs:
        pairs.append(pair)
    
    return pairs

  def __find_ligand_opposites__assign_pairs_to_axes(self,pairs):
    """
    Helper method for __find_ligand_opposites().

    Ensures the pairs within the pairs list are assigned 
      consistently with defined axes.
    I.e., the first pair should be considered the x-axis
      second pair should be considered y-axis
      third pair should be considered z-axis
    """
    # assign pairs to axes; z- then y- then x-, then flip
    ordered_pairs = []

    pairs.sort(
      reverse=True,
      key=lambda x : (
        abs(x[0].coords[2]-x[1].coords[2]),
        -1*abs(x[0].coords[1]-x[1].coords[1]),
        -1*abs(x[0].coords[0]-x[1].coords[0]))
    )
    ordered_pairs.append(pairs[0])

    pairs.sort(
      reverse=True,
      key=lambda x : (
        abs(x[0].coords[1]-x[1].coords[1]),
        -1*abs(x[0].coords[0]-x[1].coords[0]),
        -1*abs(x[0].coords[2]-x[1].coords[2]))
    )
    if str(pairs[0])!=str(ordered_pairs[0]):
      ordered_pairs.append(pairs[0])
    else:
      ordered_pairs.append(pairs[1])

    pairs.sort(
      reverse=True,
      key=lambda x : (
        abs(x[0].coords[0]-x[1].coords[0]),
        -1*abs(x[0].coords[1]-x[1].coords[1]),
        -1*abs(x[0].coords[2]-x[1].coords[2]))
    )
    if (
      (str(pairs[0])!=str(ordered_pairs[0]) )
      and (str(pairs[0]) != str(ordered_pairs[1]))
    ):
      ordered_pairs.append(pairs[0])
    elif (
      (str(pairs[1])!=str(ordered_pairs[0])) 
      and (str(pairs[1])!=str(ordered_pairs[1]))
    ):
      ordered_pairs.append(pairs[1])
    else:
      ordered_pairs.append(pairs[2])

    ordered_pairs.reverse()
    for i in [0,1,2]:
      pairs[i] = ordered_pairs[i]

  def find_ligand_opposites(self):
    """
    Returns 3 arrays each containing two pymatgen PeriodicSite objects. 
      These are the two opposite ligands in the octahedra (i.e. have a 
      bond length 180 degrees through the central atom). It does this 
      by assuming opposite bonds will have the largest angle via the 
      central atom.

    return:
      a list of shape (3,2), where the 3 are the three pairs of 
        opposite ligands and the 2 is each ligand within a pair.
    """

    #helper function obtains all pairs
    pairs = self.__find_ligand_opposites__find_all_pairs()

    #check for duplicate pairs
    self.__find_ligand_opposites__duplicate_check(pairs)
    if len(pairs) != 3: 
      raise Exception(
        "Error: __find_ligand_opposites__duplicate_check failed."
      )

    # assign pairs to axes
    self.__find_ligand_opposites__assign_pairs_to_axes(pairs)

    #sort each pair in pairs, in ascending order for assigned axis
    for i in [0,1,2]:
      local_pair = pairs[i]
      local_pair_sorted = sorted(
        local_pair,
        key=lambda x: x.coords[i],
        reverse=True
      )
      pairs[i] = local_pair_sorted

    # make sure all ligands are assigned to pairs
    all_ligands = []
    for i in pairs:
      for j in [0,1]:
        all_ligands.append(i[j])
    for i in self.ligands:
      if i not in all_ligands:
        err_string = "Error: pair finding algorithm failed. "
        err_string += "{} ligand at {} is not in any pairs.".format(
          str(i.specie),
          i.coords
        )
        print(err_string)
        print("There are {} pairs. These are:".format(len(pairs)))
        for i in pairs:
          print(i)
        raise Exception(err_string)

    return pairs

  def __CheckValidVector(self,vector_list,variable_name):
    """
    This is a helper function. it will check that a vector (i.e. 
      2D list of form [[1,0,0],[0,1,0],[0,0,1]]) is valid and output 
      appropriate error messages if not.

    arguments:
      vector_list: the potential vector (should be a list with 
        shape (3,3))
      variable_name: the string name of variable, for error outputs
    """

    if (
      (type(vector_list) !=type(False) or vector_list != False) 
      and (type(vector_list) != type(None))
    ):

      if type(vector_list) not in [type([]),type(np.array([]))]:
        raise AssertionError(
          "Error: {} is not of the correct format.".format(
            variable_name
          )
        )

      if len(vector_list) != 3:
        raise ValueError(
          "Error: {} must contain three axes".format(variable_name)
        )

      for i in [0,1,2]:

        if type(vector_list[i]) not in [type([]),type(np.array([]))]:
          raise ValueError(
            "Error: {} is not of the correct format.".format(
              variable_name
            )
          )

        if len(vector_list[i]) != 3:
          error_message = "Error: each axis in {} must contain three components.".format(
            variable_name
          )
          raise ValueError(error_message)

        for j in [0,1,2]:

          if (
            type(vector_list[i][j]) 
            not in [type(1),type(1.0),np.float64,np.int32]
          ):
            raise ValueError(
              "Error: {} components must be floats or integers but {} is {}.".format(
                variable_name,
                vector_list[i][j],
                type(vector_list[i][j])
              )
            )

          if i != j: # check axes are perpendicular
            if np.dot(vector_list[i],vector_list[j]) >= 1e-4:
              #should be zero, tolerance due to floating point errors
              raise ValueError(
                "Error: {} must be mutually orthogonal.".format(
                  variable_name
                )
              )

        if (
          vector_list[i][0]**2 
          + vector_list[i][1]**2 
          + vector_list[i][2]**2 
          == 0
        ):
          raise ValueError(
            "Error: vector {} may not have components with value zero".format(
              variable_name
            )
          )

  def __VanVleckDistortionModes_CheckInputs(
      self,
      octahedral_centre,
      specified_axes,
      automatic_rotation,
      ignore_angular_distortion,
      output_pairs,
      suppress_warnings,
      rotation_tolerance,
      omit_failed_rotations
    ):
    """
    This helper method checks the inputs to the 
      calculate_van_vleck_distortion_modes() method to make sure they 
      are valid.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      ignore_angular_distortion: if True, no rotation operations will 
        be performed, and the van Vleck modes will be calculated along 
        bond lengths without regard to their angles.
      suppress_warnings: bool. if True, will suppress warnings to 
        console.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output.
      output_pairs: bool. if True, the code gives as a return the pairs 
        list. 
    """
    if type(automatic_rotation) != type(True):
      raise ValueError("Error: automatic_rotation must be boolean.")

    if type(omit_failed_rotations)!=type(True): 
      raise ValueError("Error: omit_failed_rotations must be boolean.")
    
    if type(rotation_tolerance) not in [type(1),type(0.25)]:
      raise ValueError("Error: rotation_tolerance must be int or float.")
    
    if type(output_pairs)!=type(True):
      raise ValueError("Error: output_pairs must be boolean.")

    if type(octahedral_centre) not in [type("xyz"),type([]),type(np.array([]))]: 
      raise ValueError("Error: octahedral_centre must be string or list.")
    
    if type(suppress_warnings)!=type(True):
      raise ValueError("Error: suppress_warnings must be boolean.")

    if type(octahedral_centre) != type("xyz"):

      if len(octahedral_centre)!=3:
        raise ValueError(
          "Error: 3 coordinates are required in octahedral_centre."
        )

      for i in [0,1,2]:
        axis_coordinates = [j.coords[i] for j in self.ligands]
        if (
          type(octahedral_centre[i]) 
          not in [type(1),type(1.0),np.float64]
        ):
          err = "Error: octahedral_centre must be floats or integers. Type is {}."
          raise ValueError(
            err.format(str(type(octahedral_centre[i])))
          )

        err = "Error: octahedral_centre must be within octahedron."
        err += " octahedral_centre[{}] has value {} which is too {}. Coords are {}."
        if not (min(axis_coordinates) < octahedral_centre[i]):
          raise ValueError(
            err.format(
              str(i),
              str(octahedral_centre[i]),
              "low",
              str(axis_coordinates)
            )
          )
        if not (max(axis_coordinates) > octahedral_centre[i]):
          raise ValueError(
            err.format(
              str(i),
              str(octahedral_centre[i]),
              "large",
              str(axis_coordinates)
            )
          )
    else:
      if octahedral_centre not in ["core_atom","average_ligand_position"]:
        raise ValueError(
          "Error: octahedral_centre must by either core_atom or average_ligand_position."
        )

    if type(ignore_angular_distortion)!=type(True):
      raise ValueError(
        "Error: ignore_angular_distortion must be boolean."
      )

    if ignore_angular_distortion:
      if output_pairs:
        raise ValueError(
          "Error: output_pairs is not compatible with ignore_angular_distortion."
        )

    self.__CheckValidVector(specified_axes,"specified_axes")

  def __VanVleckDistortionModes__set_origin(
      self,
      octahedral_centre
    ):
    """
    This is a helper method for the 
      self.calculate_van_vleck_distortion_modes() methods.

    This method will determine a set of ligand coordinates about a 
      determined origin in space.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.

    return:
      a list with shape (3,2,3), containing three "pairs", where each 
        "pair" contains two lists representing the x, y, and z 
        coordinates of the two ligands in the pair, relative to the 
        origin
    """

    pairs = self.find_ligand_opposites()

    origin = self.core_site.coords.copy()

    # This allows for central coordinates to be manually chosen
    if type(octahedral_centre) in [type([]),type(np.array([]))]:
      origin=octahedral_centre

    elif octahedral_centre=="average_ligand_position":
      origin=[
        np.mean([i.coords[j] for i in self.ligands]) for j in [0,1,2]
      ]

    # First, we want all ligands to be coordinates relative to 
    #   centre of octahedron
    for i in [0,1,2]:
      for j in [0,1]:

        x = pairs[i][j].coords[0] - origin[0]
        y = pairs[i][j].coords[1] - origin[1]
        z = pairs[i][j].coords[2] - origin[2]

        pairs[i][j] = [x,y,z]

    return pairs

  def __RotateAxesAutomatic_SingleAxis(
    self,
    pairs,
    axis1,
    axis2,
  ):
    """
    This is a helper method for the 
      self.calculate_van_vleck_distortion_modes() methods.

    This method will perform an automatic rotation of all ligands 
      in the octahedron to make sure the octahedral axes are as close 
      as possible to the x-, y-, and z- axes. This is the default, the
      alternative being to manually define the axes in an argument to
      self.calculate_van_vleck_distortion_modes().

    arguments:
      pairs is a 3x2x3 list. Three pairs, each with two atoms 
        (represented by lists), each with 3 spatial coordinates. 
      axis1: integer. An axis in the plane within which rotation is 
        occurring. Must be in [0,1,2], where 0=x, 1=y, 2=z
      axis2: integer. An axis in the plane within which rotation is 
        occurring. Must be in [0,1,2], where 0=x, 1=y, 2=z

    return:
      the angle (in radians) by which the octahedron is rotated  
        within the plane.
    """
    #check good inputs
    if type(pairs) != type([]):
      raise Exception("Error: pairs must be list.")
    if any([type(i)!=type(1) for i in [axis1,axis2]]): 
      raise Exception("Error: axis1 and axis2 must be integers.")
    if (axis1 not in [0,1,2]) and (axis2 in [0,1,2]): 
      raise Exception("Error: axis1 and axis 2 must be in [0,1,2]")

    #put sites in clockwise order about axis
    sites = [ 
      pairs[axis2][1],
      pairs[axis1][1],
      pairs[axis2][0],
      pairs[axis1][0],
      ]

    #Sort sites in ascending order of angle from the vertical axis
    angles = []
    for i in sites:
      angle = 0
      if i[axis1]*i[axis2] < 0:
        angle +=  np.arctan( abs( i[axis2] / i[axis1] ) )
      elif i[axis1]*i[axis2] > 0:
        angle +=  np.arctan( abs( i[axis1] / i[axis2] ) )
      if i[axis1] > 0 and i[axis2] <= 0:
        angle += np.pi/2
      elif i[axis1] <= 0 and i[axis2] < 0:
        angle += np.pi
      elif i[axis1] < 0 and i[axis2] >= 0:
        angle += 3*np.pi/2
      angles.append(angle)

    #calculate average_angle
    true_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    wrong_angles = []
    for i in range(0,len(angles)):
      angle = angles[i]
      angle_variants = [angle-2*np.pi,angle,angle+2*np.pi]
      wrong_angle_variants = [
        true_angles[i] - j for j in angle_variants
      ]
      wrong_angle_variants.sort(key = lambda x : abs(x))
      wrong_angles.append(wrong_angle_variants[0])
    average_angle = np.average(wrong_angles)

    #perform the rotation
    rotation_angle = average_angle
    for i in [0,1,2]:
      for j in [0,1]:
        first_val = pairs[i][j][axis1]
        second_val = pairs[i][j][axis2]
        pairs[i][j][axis1] = (  
            first_val * np.cos(rotation_angle) 
          + second_val * np.sin(rotation_angle)
        )
        pairs[i][j][axis2] = (
          - first_val * np.sin(rotation_angle) 
          + second_val * np.cos(rotation_angle)
        )

    return rotation_angle

  def __VanVleckDistortionModes_RotateAxesSpecified(
      self,
      pairs,
      specified_axes
    ):
    """
    This is a helper method for the 
      self.calculate_van_vleck_distortion_modes() method.

    This method will perform a rotation of all ligands in the 
      octahedron to make the specified_axes be the x-, y-, and 
      z- axes. This will be done in three steps:
      (1) Rotate the frame of reference around the y-axis so that 
        the first vector has z component of zero
      (2) Rotate the frame of reference around the z-axis so that 
        the first vector has y component of zero
      (3) Rotate the frame of reference around the x-axis so that 
        the second vector has z component of zero

    arguments:
      pairs is a 3x2x3 list. Three pairs, each with two atoms 
        (represented by lists), each with 3 spatial coordinates.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation. 
        They must be perpendicular to one another. For example: 
        [[1,0,0],[0,1,0],[0,0,1]] or [[1,0,1],[1,0,-1],[0,1,0]]
    """

    rotational_magnitude_tolerance=GLOBAL_PARAM__ROTATION_TOLERANCE_AXES 

    rotation_magnitude = rotational_magnitude_tolerance + 1

    while rotation_magnitude > rotational_magnitude_tolerance:

      angles_squared = 0

      # perform three rotations in terms of [axis1,axis2,vector]
      for k in [[0,2,0],[0,1,0],[1,2,1]]: 
        vector = specified_axes[k[2]]
        denominator = np.sqrt(vector[k[1]] **2 + vector[k[0]] **2)

        if denominator != 0: # if denominator == 0, no rotation
          angle = np.arcsin(
            vector[k[1]] / denominator
          )

          # magnitude of rotation in this iteration can be computed
          angles_squared += angle**2

          for i in [0,1,2]:
            for j in [0,1]:
              first_val = pairs[i][j][k[0]]
              second_val = pairs[i][j][k[1]]
              pairs[i][j][k[0]] =  (
                  first_val * np.cos(angle) 
                + second_val * np.sin(angle)
              )
              pairs[i][j][k[1]] = (
                - first_val * np.sin(angle) 
                + second_val * np.cos(angle)
              )

            # Update specified_axes
            first_val =  specified_axes[i][k[0]]
            second_val = specified_axes[i][k[1]]
            specified_axes[i][k[0]] =  (
              first_val * np.cos(angle) + second_val * np.sin(angle)
            )
            specified_axes[i][k[1]] = (
              - first_val * np.sin(angle) + second_val * np.cos(angle)
            )

      rotation_magnitude = np.sqrt(angles_squared)

  def __RotateAxesAutomatic(self,pairs,max_iterations=-1):
    """
    This is a helper method for the 
      self.calculate_van_vleck_distortion_modes() methods.

    This method will perform an automatic rotation of all ligands in 
      the octahedron to make sure the octahedral axes are as close as 
      possible to the x-, y-, and z- axes. This is the default, the
      alternative being to manually define the axes in an argument to
      self.calculate_van_vleck_distortion_modes().

    arguments:
      pairs is a 3x2x3 list. Three pairs, each with two atoms 
        (represented by lists), each with 3 spatial coordinates.
      max_iterations: integer. Limits the number of iterations. Set 
        negative for no limit. Optional argument, used for debugging.
        Defaults to -1, meaning to limit.
    """

    rotational_magnitude_tolerance=GLOBAL_PARAM__ROTATION_TOLERANCE_AXES

    rotation_magnitude = rotational_magnitude_tolerance + 1

    num_iterations = 0

    while (
      (rotation_magnitude > rotational_magnitude_tolerance) 
      and (num_iterations != max_iterations)
    ):
      a1 = self.__RotateAxesAutomatic_SingleAxis(pairs,0,1) # z axis
      a2 = self.__RotateAxesAutomatic_SingleAxis(pairs,0,2) # y axis
      a3 = self.__RotateAxesAutomatic_SingleAxis(pairs,1,2) # x axis

      rotation_magnitude = np.sqrt(a1**2 + a2**2 + a3**2)

      num_iterations += 1

  def __VanVleckDistortionModes_RotateAxes(
      self,
      pairs,
      specified_axes,
      automatic_rotation
    ):
    """
    This is a helper method for the 
      self.calculate_van_vleck_distortion_modes() method.

    This method will perform a rotation of all ligands in the 
      octahedron to make sure the octahedral axes are as close as 
      possible to the x-, y-, and z- axes. The method it uses depends 
      on whether specified_axes are specified.

    arguments:
      pairs is a 3x2x3 list. Three pairs, each with two atoms 
        (represented by lists), each with 3 spatial coordinates.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
    """

    #save values to avoid changing the list beyond the code itself
    local_copy = copy.deepcopy(specified_axes)

    #perform rotation
    self.__VanVleckDistortionModes_RotateAxesSpecified(
      pairs,
      specified_axes
    )

    #rewrite specified_axes
    for i in [0,1,2]:
      specified_axes[i] = local_copy[i]

    if automatic_rotation:

      #perform automatic rotation
      self.__RotateAxesAutomatic(pairs)

    #make sure pairs ordered correctly, i.e. match perfect_coords later
    for i in [0,1,2]:
      distances = np.array(
        [abs(pairs[i][0][j]-pairs[i][1][j]) for j in [0,1,2]]
      )
      index = np.argmax(distances)
      pairs[i] = sorted(pairs[i],key=lambda x : x[index],reverse=True)

  def __calculate_van_vleck_distortion_modes(self,VanVleck_coords):
    """
    This helper method takes a multi-dimensional list of van Vleck 
      coords and calculates the van Vleck modes, as defined by Van 
      Vleck in:
        Van Vleck, J. H. "The Jahn‐Teller Effect and Crystalline Stark 
        Splitting for Clusters of the Form XY6." The Journal of 
        Chemical Physics 7.1 (1939): 72-84.
    
    arguments:
      VanVleck_coords. A list with shape (6,3), where the 6 elements
        represent the 6 ligands in the octahedron, and the 3 elements
        are the coordinates (x,y,z) of each ligand relative to the 
        centre of the octahedron. The coords must be ordered in the 
        correct way, with the first two being those assigned to the 
        x axis, the second two assigned to y axis, and the third two
        assigned to z axis. Within each two atoms, they are arranged in
        descending order within their axis.

    return:
      a list of floats, containing the calculated values of the van 
        Vleck modes Q1 to Q6, in units of Angstroms.
    """
    Q_1 = (
      VanVleck_coords[0][0]
      - VanVleck_coords[1][0]
      + VanVleck_coords[2][1]
      - VanVleck_coords[3][1]
      + VanVleck_coords[4][2]
      - VanVleck_coords[5][2]
    ) / np.sqrt(6)

    Q_2 = 0.5 * (
      VanVleck_coords[0][0]
      - VanVleck_coords[1][0]
      - VanVleck_coords[2][1]
      + VanVleck_coords[3][1]
    )

    Q_3 = ( 0.5 * (
        VanVleck_coords[0][0]
        - VanVleck_coords[1][0]
        + VanVleck_coords[2][1]
        - VanVleck_coords[3][1]
        )
      - VanVleck_coords[4][2]
      + VanVleck_coords[5][2]
    ) / np.sqrt(3)

    Q_4 = 0.5 * (
      VanVleck_coords[0][1] - VanVleck_coords[1][1] 
      + VanVleck_coords[2][0] - VanVleck_coords[3][0]
    )

    Q_5 = 0.5 * (
      VanVleck_coords[0][2] - VanVleck_coords[1][2] 
      + VanVleck_coords[4][0] - VanVleck_coords[5][0]
    )

    Q_6 = 0.5 * (
      VanVleck_coords[2][2] - VanVleck_coords[3][2] 
      + VanVleck_coords[4][1] - VanVleck_coords[5][1]
    )

    return [Q_1,Q_2,Q_3,Q_4,Q_5,Q_6]
  
  def __VanVleckDistortionModes__reorder_pair(
      self,
      pairs,
      ordering_axes,
      priority
    ):
    """
    This helper method for __VanVleckDistortionModes__reorder_pairs 
      will reorder a certain set of pairs based on ordering_axes, with 
      the priority list setting the priority of each element in 
      ordering_axes.

    arguments:
      pairs is a 3x2x3 list. Three pairs, each with two atoms 
        (represented by lists), each with 3 spatial coordinates.
      ordering_axes is a list of shape (3,3). Each sub-list is a 
        vector, and ligands within each pair are sorted according to
        their position along each vector, with the vectors assigned
        a "priority"
      priority: a list of length 3, which are three integers 0,1,and 2
        which determine the order in ordering_axes by which the vectors
        are prioritised for ordering the ligands.
    """
    #check inputs
    if not type(priority) in [type([]),type(np.array)]:
      raise ValueError("Error: priority list must be list.")
    if not type(ordering_axes) in [type([]),type(np.array)]: 
      raise ValueError("Error: ordering_axes must be list or array.")
    if not all([type(i)==type(1) for i in priority]):
      raise ValueError("Error: priority list must be integers.")
    if len(priority)!=3:
      raise ValueError("Error: priority list must be 3 integers.")
    if any([priority[i] not in [0,1,2] for i in [0,1,2]]):
      raise ValueError(
        "Error: priority must be list of indices of ordering_axes."
      )
    if not len(set(priority))==3:
      raise ValueError(
        "Error: all elements in priority list must be unique."
      )

    #reorder pairs by length along ordering_axes[priority[0]], 
    #   then priority[1], then priority[2]
    pairs.sort( 
      reverse=True,
      key=lambda x : (
        np.sqrt(
          np.sum(
            self.__find_projection_of_a_vector(
              np.array(x[0])-np.array(x[1]),
              ordering_axes[priority[0]]
            )**2
          )
        ),
        np.sqrt(
          np.sum(
            self.__find_projection_of_a_vector(
              np.array(x[0])-np.array(x[1]),
              ordering_axes[priority[1]]
            )**2
          )
        ),
        np.sqrt(
          np.sum(
            self.__find_projection_of_a_vector(
              np.array(x[0])-np.array(x[1]),
              ordering_axes[priority[2]]
            )**2
          )
        )
      )
    )

    #make sure each pair is aligned along axis
    for i in [0,1,2]:
      if (
        np.dot(
          self.__find_projection_of_a_vector(
            np.array(pairs[i][1])-np.array(pairs[i][0]),
            ordering_axes[priority[0]]
          ),
          ordering_axes[priority[0]]
        ) < 0
      ): # make sure each pair is sorted by length along vector
        pairs[i] = [pairs[i][1],pairs[i][0]]

  
  def __VanVleckDistortionModes__reorder_pairs(
      self,
      pairs,
      specified_axes
    ):
    """
    This helper method for calculate_van_vleck_distortion_modes 
      determines the order of pairs such that they are ordered on 
      the basis of the distance between ligands in the pair projected 
      on each of the three axes.
    The axes are given ascending order of preference.

    arguments:
      pairs is a 3x2x3 list. Three pairs, each with two atoms 
        (represented by lists), each with 3 spatial coordinates.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation. 
        They must be perpendicular to one another. For example: 
        [[1,0,0],[0,1,0],[0,0,1]] or [[1,0,1],[1,0,-1],[0,1,0]]
    """
    # new pair list
    ordered_pairs = []

    #assign a pair to the z axis
    self.__VanVleckDistortionModes__reorder_pair(
      pairs,
      specified_axes,
      [2,1,0]
    )
    ordered_pairs.append(pairs[0])

    #assign a pair to the y axis
    self.__VanVleckDistortionModes__reorder_pair(
      pairs,
      specified_axes,
      [1,0,2]
    )
    if (
      str(ordered_pairs[0]) 
      not in [str(pairs[0]),str([pairs[0][1],pairs[0][0]])]
    ):
      append_num = 0 
    else:
      append_num = 1
    ordered_pairs.append(pairs[append_num])

    #assign a pair to the x axis
    self.__VanVleckDistortionModes__reorder_pair(
      pairs,
      specified_axes,
      [0,2,1]
    )
    if (
      (
        str(ordered_pairs[0]) 
        not in [str(pairs[0]),str([pairs[0][1],pairs[0][0]])]
      )
      and (
        str(ordered_pairs[1]) 
        not in [str(pairs[0]),str([pairs[0][1],pairs[0][0]])]
      )
    ):
      append_num = 0
    elif (
      (
        str(ordered_pairs[0]) 
        not in [str(pairs[1]),str([pairs[1][1],pairs[1][0]])]
      )
      and (
        str(ordered_pairs[1]) 
        not in [str(pairs[1]),str([pairs[1][1],pairs[1][0]])]
      )
    ):
      append_num = 1
    else:
      append_num = 2
    ordered_pairs.append(pairs[append_num])

    #update main pairs list 
    ordered_pairs.reverse()
    for i in [0,1,2]:
      pairs[i] = ordered_pairs[i]

  def __VanVleckDistortionModes__fix_axes_cross_product(
    self,
    specified_axes
  ):
    """
    This helper method for Van Vleck calculation makes sure that the
      axes are oriented correctly relative to one another. 
      Specifically, that the cross product of the first two components 
      is parallel to the third component

    arguments:
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation. 
        They must be perpendicular to one another. For example: 
        [[1,0,0],[0,1,0],[0,0,1]] or [[1,0,1],[1,0,-1],[0,1,0]]
    """
    if (
      np.dot(
        np.cross(
          specified_axes[0],
          specified_axes[1]
        ),
        specified_axes[2]
      )
    ) < 0:
      specified_axes[2] = -1 * np.array(specified_axes[2])

  def __calculate_van_vleck_distortion_modes__set_pairs(
        self,
        octahedral_centre,
        specified_axes,
        suppress_warnings,
        automatic_rotation,
        ignore_angular_distortion,
        rotation_tolerance, 
        omit_failed_rotations,
      ):
    """
    This helper method assists in the calculation of the van 
      Vleck modes.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      ignore_angular_distortion: if True, no rotation operations will 
        be performed, and the van Vleck modes will be calculated along 
        bond lengths without regard to their angles. 
      suppress_warnings: bool. if True, will suppress warnings to 
        console.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output.
    """

    # copy specified_axes
    specified_axes = copy.deepcopy(specified_axes)

    # Obtain a set of ligand coordinates about a determined origin.
    pairs = self.__VanVleckDistortionModes__set_origin(
      octahedral_centre
    )

    # make sure the cross product of the first and second axis is 
    #   parallel to the third axis
    self.__VanVleckDistortionModes__fix_axes_cross_product(
      specified_axes
    ) 

    #choose order of pairs within the list such that they correspond 
    #   to longest length along the vectors in specified_axes 
    self.__VanVleckDistortionModes__reorder_pairs(
      pairs,
      specified_axes
    )   

    # Next we want to rotate the frame of reference about axes so 
    #   that the octahedral axes match the x-, y-, z- spatial axes
    if not ignore_angular_distortion:
      self.__VanVleckDistortionModes_RotateAxes(
        pairs,
        specified_axes,
        automatic_rotation
      )
    else:
      #if ignore_angular_distortion there is still an initial rotation
      self.__VanVleckDistortionModes_RotateAxes(
        pairs,
        specified_axes,
        False
      )

    #warning, only if not ignore_angular_distortion
    successful_rotation = True
    if not ignore_angular_distortion:
      if suppress_warnings==False:
        for i in [0,1,2]:
          pair = pairs[i]
          for j in [0,1]:
            indices = [k for k in [0,1,2] if k != i]
            mismatched_magnitude  = (
              np.sqrt(pair[j][indices[0]]**2 + pair[j][indices[1]]**2) 
              / pair[j][i]
            )
            if mismatched_magnitude > rotation_tolerance:
              successful_rotation = False
              msg = "Warning: automatic axis-finding algorithm for octahedron centred on {} site may have failed."
              msg = msg.format(str(self.core_site.coords))
              msg += "\nMismatched magnitude for site {} in pair {} has value {}, exceeding defined limit of {}.".format(
                j, i,
                mismatched_magnitude,
                rotation_tolerance
              )
              msg += "\nautomatic_rotation is {} and specified_axes is {}.".format(
                automatic_rotation,
                specified_axes
              )
              if (
                str(specified_axes)==str([[1,0,0],[0,1,0],[0,0,1]])
              ):
                msg += "\nRecommend you use specified_axes argument."
              else: # if axes are given
                msg += "\nRecommend you try different specified_axes."
              if omit_failed_rotations:
                msg += "\nAs omit_failed_rotations==True, this site will return None "
                msg += "for all calculations."
              warnings.warn(msg)

    return pairs , successful_rotation

  def calculate_van_vleck_distortion_modes(
        self,
        octahedral_centre="core_atom",
        specified_axes=[[1,0,0],[0,1,0],[0,0,1]] ,
        suppress_warnings=False,
        automatic_rotation=True,
        output_pairs=False,
        ignore_angular_distortion=False,
        rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
        omit_failed_rotations=False,
      ):
    """
    This method calculates the magnitude of the Q_1 to Q_6 distortion 
      modes, as defined by Van Vleck:
        Van Vleck, J. H. "The Jahn‐Teller Effect and Crystalline Stark
        Splitting for Clusters of the Form XY6." The Journal of 
        Chemical Physics 7.1 (1939): 72-84.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      ignore_angular_distortion: if True, no rotation operations will 
        be performed, and the van Vleck modes will be calculated along 
        bond lengths without regard to their angles. 
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      output_pairs: bool. if True, the code gives as a return the pairs 
        list. Defaults to False.

    return:
      -a list containing the 6 Van Vleck modes
      -[if output_pairs]: a 2D list containing the coordinates of all sites after axis transformation
    """

    # Check inputs are valid
    self.__VanVleckDistortionModes_CheckInputs(
      octahedral_centre,
      specified_axes,
      automatic_rotation,
      ignore_angular_distortion,
      output_pairs,
      suppress_warnings,
      rotation_tolerance,
      omit_failed_rotations,
    )

    # prepare pairs list
    pairs , successful_rotation = self.__calculate_van_vleck_distortion_modes__set_pairs(
      octahedral_centre=octahedral_centre,
      specified_axes=specified_axes,
      suppress_warnings=suppress_warnings,
      automatic_rotation=automatic_rotation,
      ignore_angular_distortion=ignore_angular_distortion,
      rotation_tolerance=rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    #Calculate average bond length
    ideal_bond_length = self.calculate_average_ligand_bond_length(
      octahedral_centre=octahedral_centre
    )

    perfect_coords = np.array([
      [-1*ideal_bond_length,0,0], [ideal_bond_length,0,0],
      [0,-1*ideal_bond_length,0], [0,ideal_bond_length,0],
      [0,0,-1*ideal_bond_length], [0,0,ideal_bond_length]
    ])

    # calculate using cartesian basis
    if not ignore_angular_distortion: 
      site_coords = np.array([
        [pairs[0][1][0],pairs[0][1][1],pairs[0][1][2]],
        [pairs[0][0][0],pairs[0][0][1],pairs[0][0][2]],

        [pairs[1][1][0],pairs[1][1][1],pairs[1][1][2]],
        [pairs[1][0][0],pairs[1][0][1],pairs[1][0][2]],

        [pairs[2][1][0],pairs[2][1][1],pairs[2][1][2]],
        [pairs[2][0][0],pairs[2][0][1],pairs[2][0][2]]
      ])
    
    # or calculate using bond lengths
    else: 
      sites = [
        pairs[0][1], pairs[0][0], pairs[1][1],
        pairs[1][0], pairs[2][1], pairs[2][0]
      ]
      magnitudes = [np.sqrt(i[0]**2+i[1]**2+i[2]**2) for i in sites]
      site_coords = np.array([
        [-magnitudes[0],0,0], [magnitudes[1],0,0],
        [0,-magnitudes[2],0],[0,magnitudes[3],0],
        [0,0,-magnitudes[4]], [0,0,magnitudes[5]]
      ])

    VanVleck_coords = site_coords-perfect_coords

    van_vleck_modes = self.__calculate_van_vleck_distortion_modes(
      VanVleck_coords
    )

    if ((not successful_rotation) and (omit_failed_rotations)):
      van_vleck_modes = [None for _ in van_vleck_modes]

    if output_pairs:
      return van_vleck_modes , pairs
    else:
      return van_vleck_modes

  def calculate_degenerate_Q3_van_vleck_modes(
        self,
        octahedral_centre="core_atom",
        specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
        suppress_warnings=False,
        automatic_rotation=True,
        output_pairs = False,
        ignore_angular_distortion=False,
        rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
        omit_failed_rotations=False,
      ):
    """
    This method outputs the magnitude of the Q_3 distortion mode 
      and its two symmetrically identical linear combinations of Q_2 
      and Q_3:
        -1/2 Q_3 ± √3/2 Q_2
      where Q_2 and Q_3 are as defined by Van Vleck:
        Van Vleck, J. H. "The Jahn‐Teller Effect and Crystalline Stark
          Splitting for Clusters of the Form XY6." The Journal of 
          Chemical Physics 7.1 (1939): 72-84.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      ignore_angular_distortion: if True, no rotation operations will 
        be performed, and the van Vleck modes will be calculated along 
        bond lengths without regard to their angles. 
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. 
      output_pairs: bool. if True, the code gives as a return the pairs 
        list. Defaults to False.
    """

    #obtain Van Vleck modes
    if output_pairs:
      VanVleck_modes, pairs = self.calculate_van_vleck_distortion_modes(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        suppress_warnings=suppress_warnings,
        automatic_rotation=automatic_rotation,
        output_pairs = True,
        ignore_angular_distortion = ignore_angular_distortion,
        rotation_tolerance = rotation_tolerance,
        omit_failed_rotations = omit_failed_rotations,
      )
    else:
      VanVleck_modes = self.calculate_van_vleck_distortion_modes(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        suppress_warnings=suppress_warnings,
        automatic_rotation=automatic_rotation,
        output_pairs = False,
        ignore_angular_distortion = ignore_angular_distortion,
        rotation_tolerance = rotation_tolerance,
      )

    modes = [
      VanVleck_modes[2],#Q_3
      (
        - 0.5 * VanVleck_modes[2] 
        + np.sqrt(3) * 0.5 * VanVleck_modes[1]
      ), # -1/2 Q_3 + √3/2 Q_2
      (
        - 0.5 * VanVleck_modes[2]  
        - np.sqrt(3) * 0.5 * VanVleck_modes[1]
      ), # -1/2 Q_3 - √3/2 Q_2
    ]

    if output_pairs:
      return modes, pairs
    else:
      return modes


  def calculate_van_vleck_jahn_teller_params(
        self,
        octahedral_centre="core_atom",
        specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
        degrees=False,
        suppress_warnings=False,
        automatic_rotation=True,
        output_pairs=False,
        ignore_angular_distortion=False,
        rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
        omit_failed_rotations=False,
      ):
    """
    This method outputs sqrt(Q_3^2 + Q_2^2) and arctan(Q_2/Q_3), which 
      can be used to generate a polar plot as in, for example:
        Zhou, J-S., et al. "Jahn–Teller distortion in perovskite 
        KCuF3 under high pressure."
        Journal of Fluorine Chemistry 132.12 (2011): 1117-1121.
        Figure 4.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      ignore_angular_distortion: if True, no rotation operations will 
        be performed, and the van Vleck modes will be calculated along 
        bond lengths without regard to their angles. 
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      output_pairs: bool. if True, the code gives as a return the pairs 
        list. Defaults to False.
      degrees: bool. Defaults to False. If True, phi is returned in 
        degrees, otherwise it is in radians.

    Return:
      -a list containing the params: magnitude and angle
      - [if output_pairs]: a list containing the atomic positions 
        within a list with shape (3,2,3)
    """

    #obtain Van Vleck modes
    if output_pairs:
      VanVleck_modes, pairs = self.calculate_van_vleck_distortion_modes(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        suppress_warnings=suppress_warnings,
        automatic_rotation=automatic_rotation,
        output_pairs = True,
        ignore_angular_distortion = ignore_angular_distortion,
        rotation_tolerance=rotation_tolerance,
        omit_failed_rotations=omit_failed_rotations,
      )
    else:
      VanVleck_modes = self.calculate_van_vleck_distortion_modes(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        suppress_warnings=suppress_warnings,
        automatic_rotation=automatic_rotation,
        output_pairs = False,
        ignore_angular_distortion = ignore_angular_distortion,
        rotation_tolerance=rotation_tolerance,
        omit_failed_rotations=omit_failed_rotations,
      )
    
    if not all([type(i)==type(None) for i in VanVleck_modes]):


      Q2 = VanVleck_modes[1]
      Q3 = VanVleck_modes[2]

      #calculate magnitude
      mag = np.sqrt(Q2**2 + Q3**2)

      #calculate angle in radians
      angle = 0
      if Q2*Q3 < 0:
        angle +=  np.arctan( abs( Q3 / Q2 ) )
      elif Q2*Q3 > 0:
        angle +=  np.arctan( abs( Q2 / Q3 ) )
      if Q2 > 0 and Q3 <= 0:
        angle += np.pi/2
      elif Q2 <= 0 and Q3 < 0:
        angle += np.pi
      elif Q2 < 0 and Q3 >= 0:
        angle += 3*np.pi/2

      #potentially convert to degrees
      if degrees:
        angle = np.degrees(angle)

      #output
      if output_pairs:
        return [ mag, angle ], pairs
      else:
        return [ mag, angle ]
    
    else : # if rotation failed
      if output_pairs:
        return [ None, None ], pairs
      else:
        return [ None, None ]
    
  def output_sites_for_van_vleck(
        self,
        octahedral_centre="core_atom",
        specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
        automatic_rotation=True,
        suppress_warnings=False,
        rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
        omit_failed_rotations=False,
      ):
    """
    This method performs rotation of the ligands around a central point 
      and returns their positions. The purpose of this is for checking 
      that the ligand rotation for van Vleck calculation is working
      correctly.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
    """
    # Check inputs are valid
    self.__VanVleckDistortionModes_CheckInputs(
      octahedral_centre,
      specified_axes,
      automatic_rotation=automatic_rotation,
      ignore_angular_distortion=False,
      output_pairs=False,
      suppress_warnings=False,
      rotation_tolerance = rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    # prepare pairs list
    pairs , _ = self.__calculate_van_vleck_distortion_modes__set_pairs(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        suppress_warnings=suppress_warnings,
        automatic_rotation=automatic_rotation,
        ignore_angular_distortion=False,
        rotation_tolerance=rotation_tolerance,
        omit_failed_rotations=omit_failed_rotations,
      )

    return pairs

  def visualise_sites_for_van_vleck(
        self,
        octahedral_centre="core_atom",
        specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
        automatic_rotation=True,
        output_pairs=False,
        rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      ):
    """
    This method performs rotation of the ligands around a central point
      and plots their positions. The purpose of this is for checking 
      that the ligand rotation for van Vleck calculation is working
      correctly.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      output_pairs: if True, the code gives as a return the pairs 
        list. Defaults to False.

    Action:
      plots 3D interactive plot showing the positions of the sites, to 
        check whether they are lose
    """
    #obtain pairs
    pairs = self.output_sites_for_van_vleck(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        automatic_rotation=automatic_rotation,
        rotation_tolerance=rotation_tolerance,
      )

    #plot these
    _ = self.__visualise_pairs(
      [[ np.array(i[0]),np.array(i[1])] for i in pairs ]
    )

    #output pairs:
    if output_pairs: 
      return pairs

  def output_and_visualise_sites_for_van_vleck(
        self,
        octahedral_centre="core_atom",
        specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
        automatic_rotation=True,
        rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      ):
    """
    This method performs rotation of the ligands around a central 
      point and plots their positions. The purpose of this is for 
      checking that the ligand rotation for van Vleck calculation 
      is working correctly. It also returns a Python list with 
      shape (3,2,3) containing the coordinates of each site after 
      the rotation.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.

    Action:
      plots 3D interactive plot showing the positions of the sites, to 
        check whether they are lose
    """
    #obtain pairs
    pairs = self.visualise_sites_for_van_vleck(
        octahedral_centre=octahedral_centre,
        specified_axes=specified_axes,
        automatic_rotation=automatic_rotation,
        output_pairs=True,
        rotation_tolerance=rotation_tolerance,
      )

    #return these
    return pairs
  
  def __calculate_angular_shear_modes(
      self,
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    ):
    """
    This method calculates the angular shear modes, Delta ab, 
      Delta ac, and Delta bc.
    
    arguments:
      Delta_ab_angles: list of 4 floats. The angles between the 
        assigned axis for each ligand, and the line from the 
        origin to the ligand, within the ab-plane. Can be in units
        of radians or degrees.
      Delta_ac_angles: list of 4 floats. The angles between the 
        assigned axis for each ligand, and the line from the 
        origin to the ligand, within the ac-plane. Can be in units
        of radians or degrees.
      Delta_bc_angles: list of 4 floats. The angles between the 
        assigned axis for each ligand, and the line from the 
        origin to the ligand, within the bc-plane. Can be in units
        of radians or degrees.

    return:
      a list of 3 elements, the shear parameters for the ab, ac,
        and bc planes respectively. Can be units of radians or degrees 
        depending on the values supplied to the method.
    """

    return [
      angles[0]-angles[1]+angles[2]-angles[3] for angles in [
        Delta_ab_angles,
        Delta_ac_angles,
        Delta_bc_angles
      ]
    ]
  
  def __calculate_angular_antishear_modes(
    self,
    Delta_ab_angles,
    Delta_ac_angles,
    Delta_bc_angles
  ):
    """
    This method calculates the angular anti-shear modes, Delta' ab, 
      Delta' ac, and Delta' bc.

    arguments:
      Delta_ab_angles: list of 4 floats. The angles between the 
        assigned axis for each ligand, and the line from the 
        origin to the ligand, within the ab-plane. Can be in units
        of radians or degrees.
      Delta_ac_angles: list of 4 floats. The angles between the 
        assigned axis for each ligand, and the line from the 
        origin to the ligand, within the ac-plane. Can be in units
        of radians or degrees.
      Delta_bc_angles: list of 4 floats. The angles between the 
        assigned axis for each ligand, and the line from the 
        origin to the ligand, within the bc-plane. Can be in units
        of radians or degrees.

    return:
      a list of 3 elements, the antishear parameters for the ab, ac,
        and bc planes respectively. Can be units of radians or degrees 
        depending on the values supplied to the method.
    """

    return [
      angles[0]+angles[1]-angles[2]-angles[3] for angles in [
        Delta_ab_angles,
        Delta_ac_angles,
        Delta_bc_angles
      ]
    ]
  
  def calculate_angular_shear_modes(
      self,
      octahedral_centre="core_atom",
      specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
      suppress_warnings=False,
      automatic_rotation=True,
      rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      omit_failed_rotations=False,
      degrees=False
    ):

    """
    This method calculates the angular shear modes, Delta ab, 
      Delta ac, and Delta bc.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      degrees: bool. If True, returns units of degrees, otherwise 
        radians. Defaults to False.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.

    return:
      list: a list of the three angular shear values, in units of
        degrees or radians depending on user-supplied arguments.
    """

    pairs = self.output_sites_for_van_vleck(
      octahedral_centre=octahedral_centre,
      specified_axes=specified_axes,
      automatic_rotation=automatic_rotation,
      suppress_warnings=suppress_warnings,
      rotation_tolerance=rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    Delta_ab_sites = [pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]]
    Delta_ac_sites = [pairs[2][0],pairs[0][0],pairs[2][1],pairs[0][1]]
    Delta_bc_sites = [pairs[1][0],pairs[2][0],pairs[1][1],pairs[2][1]]

    Delta_ab_angles = [
      np.arctan(Delta_ab_sites[0][1]/Delta_ab_sites[0][0]),
      -1*np.arctan(Delta_ab_sites[1][0]/Delta_ab_sites[1][1]),
      np.arctan(Delta_ab_sites[2][1]/Delta_ab_sites[2][0]),
      -1*np.arctan(Delta_ab_sites[3][0]/Delta_ab_sites[3][1]),
    ]

    Delta_ac_angles = [
      np.arctan(Delta_ac_sites[0][0]/Delta_ac_sites[0][2]),
      -1*np.arctan(Delta_ac_sites[1][2]/Delta_ac_sites[1][0]),
      np.arctan(Delta_ac_sites[2][0]/Delta_ac_sites[2][2]),
      -1*np.arctan(Delta_ac_sites[3][2]/Delta_ac_sites[3][0]),
    ]

    Delta_bc_angles = [
      np.arctan(Delta_bc_sites[0][2]/Delta_bc_sites[0][1]),
      -1*np.arctan(Delta_bc_sites[1][1]/Delta_bc_sites[1][2]),
      np.arctan(Delta_bc_sites[2][2]/Delta_bc_sites[2][1]),
      -1*np.arctan(Delta_bc_sites[3][1]/Delta_bc_sites[3][2]),
    ]
    
    if degrees:
      for i in [0,1,2,3]:
        Delta_ab_angles[i] = np.degrees(Delta_ab_angles[i])
        Delta_ac_angles[i] = np.degrees(Delta_ac_angles[i])
        Delta_bc_angles[i] = np.degrees(Delta_bc_angles[i])

    output = self.__calculate_angular_shear_modes(
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    )

    return output
  
  def calculate_angular_antishear_modes(
      self,
      octahedral_centre="core_atom",
      specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
      suppress_warnings=False,
      automatic_rotation=True,
      rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      omit_failed_rotations=False,
      degrees=False,
    ):
    """
    This method calculates the angular anti-shear modes, Delta' ab, 
      Delta' ac, and Delta' bc.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      degrees: bool. If True, returns units of degrees, otherwise 
        radians. Defaults to False.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.

    return:
      list: a list of the three angular anti-shear values, in units of
        degrees or radians depending on user-supplied arguments.
    """

    pairs = self.output_sites_for_van_vleck(
      octahedral_centre=octahedral_centre,
      specified_axes=specified_axes,
      automatic_rotation=automatic_rotation,
      suppress_warnings=suppress_warnings,
      rotation_tolerance=rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    Delta_ab_sites = [pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]]
    Delta_ac_sites = [pairs[2][0],pairs[0][0],pairs[2][1],pairs[0][1]]
    Delta_bc_sites = [pairs[1][0],pairs[2][0],pairs[1][1],pairs[2][1]]

    Delta_ab_angles = [
      np.arctan(Delta_ab_sites[0][1]/Delta_ab_sites[0][0]),
      -1*np.arctan(Delta_ab_sites[1][0]/Delta_ab_sites[1][1]),
      np.arctan(Delta_ab_sites[2][1]/Delta_ab_sites[2][0]),
      -1*np.arctan(Delta_ab_sites[3][0]/Delta_ab_sites[3][1]),
    ]

    Delta_ac_angles = [
      np.arctan(Delta_ac_sites[0][0]/Delta_ac_sites[0][2]),
      -1*np.arctan(Delta_ac_sites[1][2]/Delta_ac_sites[1][0]),
      np.arctan(Delta_ac_sites[2][0]/Delta_ac_sites[2][2]),
      -1*np.arctan(Delta_ac_sites[3][2]/Delta_ac_sites[3][0]),
    ]

    Delta_bc_angles = [
      np.arctan(Delta_bc_sites[0][2]/Delta_bc_sites[0][1]),
      -1*np.arctan(Delta_bc_sites[1][1]/Delta_bc_sites[1][2]),
      np.arctan(Delta_bc_sites[2][2]/Delta_bc_sites[2][1]),
      -1*np.arctan(Delta_bc_sites[3][1]/Delta_bc_sites[3][2]),
    ]

    if degrees:
      for i in [0,1,2,3]:
        Delta_ab_angles[i] = np.degrees(Delta_ab_angles[i])
        Delta_ac_angles[i] = np.degrees(Delta_ac_angles[i])
        Delta_bc_angles[i] = np.degrees(Delta_bc_angles[i])

    output = self.__calculate_angular_antishear_modes(
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    )

    return output
  
  def calculate_angular_shear_magnitude(
      self,
      octahedral_centre="core_atom",
      specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
      suppress_warnings=False,
      automatic_rotation=True,
      rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      omit_failed_rotations=False,
    ):
    """
    This method calculates the Delta_shear parameter.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.

    return:
      float: the square root of the sum of the angular shear 
        parameters for the three planes. In units of
        degrees or radians depending on user-supplied arguments.
    """
    pairs = self.output_sites_for_van_vleck(
      octahedral_centre=octahedral_centre,
      specified_axes=specified_axes,
      automatic_rotation=automatic_rotation,
      suppress_warnings=suppress_warnings,
      rotation_tolerance=rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    Delta_ab_sites = [pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]]
    Delta_ac_sites = [pairs[2][0],pairs[0][0],pairs[2][1],pairs[0][1]]
    Delta_bc_sites = [pairs[1][0],pairs[2][0],pairs[1][1],pairs[2][1]]

    Delta_ab_angles = [
      np.arctan(Delta_ab_sites[0][1]/Delta_ab_sites[0][0]),
      -1*np.arctan(Delta_ab_sites[1][0]/Delta_ab_sites[1][1]),
      np.arctan(Delta_ab_sites[2][1]/Delta_ab_sites[2][0]),
      -1*np.arctan(Delta_ab_sites[3][0]/Delta_ab_sites[3][1]),
    ]

    Delta_ac_angles = [
      np.arctan(Delta_ac_sites[0][0]/Delta_ac_sites[0][2]),
      -1*np.arctan(Delta_ac_sites[1][2]/Delta_ac_sites[1][0]),
      np.arctan(Delta_ac_sites[2][0]/Delta_ac_sites[2][2]),
      -1*np.arctan(Delta_ac_sites[3][2]/Delta_ac_sites[3][0]),
    ]

    Delta_bc_angles = [
      np.arctan(Delta_bc_sites[0][2]/Delta_bc_sites[0][1]),
      -1*np.arctan(Delta_bc_sites[1][1]/Delta_bc_sites[1][2]),
      np.arctan(Delta_bc_sites[2][2]/Delta_bc_sites[2][1]),
      -1*np.arctan(Delta_bc_sites[3][1]/Delta_bc_sites[3][2]),
    ]

    shear_values = self.__calculate_angular_shear_modes(
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    )

    return np.sqrt(np.sum(np.array(shear_values)**2))
  
  def calculate_angular_antishear_magnitude(
      self,
      octahedral_centre="core_atom",
      specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
      suppress_warnings=False,
      automatic_rotation=True,
      rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      omit_failed_rotations=False,
    ):
    """
    This method calculates the Delta_antishear parameter.

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.

    return:
      float: the square root of the sum of the angular antishear 
        parameters for the three planes. In units of
        degrees or radians depending on user-supplied arguments.
    """
    pairs = self.output_sites_for_van_vleck(
      octahedral_centre=octahedral_centre,
      specified_axes=specified_axes,
      automatic_rotation=automatic_rotation,
      suppress_warnings=suppress_warnings,
      rotation_tolerance=rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    Delta_ab_sites = [pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]]
    Delta_ac_sites = [pairs[2][0],pairs[0][0],pairs[2][1],pairs[0][1]]
    Delta_bc_sites = [pairs[1][0],pairs[2][0],pairs[1][1],pairs[2][1]]

    Delta_ab_angles = [
      np.arctan(Delta_ab_sites[0][1]/Delta_ab_sites[0][0]),
      -1*np.arctan(Delta_ab_sites[1][0]/Delta_ab_sites[1][1]),
      np.arctan(Delta_ab_sites[2][1]/Delta_ab_sites[2][0]),
      -1*np.arctan(Delta_ab_sites[3][0]/Delta_ab_sites[3][1]),
    ]

    Delta_ac_angles = [
      np.arctan(Delta_ac_sites[0][0]/Delta_ac_sites[0][2]),
      -1*np.arctan(Delta_ac_sites[1][2]/Delta_ac_sites[1][0]),
      np.arctan(Delta_ac_sites[2][0]/Delta_ac_sites[2][2]),
      -1*np.arctan(Delta_ac_sites[3][2]/Delta_ac_sites[3][0]),
    ]

    Delta_bc_angles = [
      np.arctan(Delta_bc_sites[0][2]/Delta_bc_sites[0][1]),
      -1*np.arctan(Delta_bc_sites[1][1]/Delta_bc_sites[1][2]),
      np.arctan(Delta_bc_sites[2][2]/Delta_bc_sites[2][1]),
      -1*np.arctan(Delta_bc_sites[3][1]/Delta_bc_sites[3][2]),
    ]

    antishear_values = self.__calculate_angular_antishear_modes(
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    )

    return np.sqrt(np.sum(np.array(antishear_values)**2))

  def calculate_shear_fraction_angular_distortion(
      self,
      octahedral_centre="core_atom",
      specified_axes=[[1,0,0],[0,1,0],[0,0,1]],
      suppress_warnings=False,
      automatic_rotation=True,
      rotation_tolerance=GLOBAL_PARAM__FAILED_ROTATION_THRESHOLD,
      omit_failed_rotations=False,
    ):
    """
    This method calculates a parameter which represents the fraction 
      of angular distortion which stems from octahedral shear

    arguments:
      octahedral_centre (string or list): if "core_atom", the centre 
        of the octahedron is the position of the core atom. If 
        "average_ligand_position" then the octahedral centre
        is the average position of the ligands. If list, must be 
        a list of three floats which are the coordinates to be taken
        as the centre of the octahedron.
      specified_axes must be a list containing three lists, with each 
        sub-list containing three elements. These are three vectors 
        which will be set as the axes for the Van Vleck calculation.  
        They must be perpendicular to eachother.
        For example: [[1,0,0],[0,1,0],[0,0,1]] 
          or [[1,0,1],[1,0,-1],[0,1,0]]
      automatic_rotation: bool. Defaults to True. If True, the 
        automated rotation of the octahedron will occur. If False, it
        will not.
      rotation_tolerance: float, in units of Angstroms. If an atom is
        more than this distance from its assigned axis, the rotation
        is considered to have failed.
      omit_failed_rotations: bool. If True, when rotation fails, 
        returns only NoneType output. Defaults to False.
      suppress_warnings: bool. if True, will suppress warnings to 
        console. Defaults to False.
    
    return:
      float: the shear fraction, eta
    """
    pairs = self.output_sites_for_van_vleck(
      octahedral_centre=octahedral_centre,
      specified_axes=specified_axes,
      automatic_rotation=automatic_rotation,
      suppress_warnings=suppress_warnings,
      rotation_tolerance=rotation_tolerance,
      omit_failed_rotations=omit_failed_rotations,
    )

    Delta_ab_sites = [pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]]
    Delta_ac_sites = [pairs[2][0],pairs[0][0],pairs[2][1],pairs[0][1]]
    Delta_bc_sites = [pairs[1][0],pairs[2][0],pairs[1][1],pairs[2][1]]

    Delta_ab_angles = [
      np.arctan(Delta_ab_sites[0][1]/Delta_ab_sites[0][0]),
      -1*np.arctan(Delta_ab_sites[1][0]/Delta_ab_sites[1][1]),
      np.arctan(Delta_ab_sites[2][1]/Delta_ab_sites[2][0]),
      -1*np.arctan(Delta_ab_sites[3][0]/Delta_ab_sites[3][1]),
    ]

    Delta_ac_angles = [
      np.arctan(Delta_ac_sites[0][0]/Delta_ac_sites[0][2]),
      -1*np.arctan(Delta_ac_sites[1][2]/Delta_ac_sites[1][0]),
      np.arctan(Delta_ac_sites[2][0]/Delta_ac_sites[2][2]),
      -1*np.arctan(Delta_ac_sites[3][2]/Delta_ac_sites[3][0]),
    ]

    Delta_bc_angles = [
      np.arctan(Delta_bc_sites[0][2]/Delta_bc_sites[0][1]),
      -1*np.arctan(Delta_bc_sites[1][1]/Delta_bc_sites[1][2]),
      np.arctan(Delta_bc_sites[2][2]/Delta_bc_sites[2][1]),
      -1*np.arctan(Delta_bc_sites[3][1]/Delta_bc_sites[3][2]),
    ]

    shear_values = self.__calculate_angular_shear_modes(
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    )

    antishear_values = self.__calculate_angular_antishear_modes(
      Delta_ab_angles,
      Delta_ac_angles,
      Delta_bc_angles
    )

    Delta_shear_square = np.sum(np.array(shear_values)**2)
    Delta_antishear_square = np.sum(np.array(antishear_values)**2)

    return (
      Delta_shear_square 
      / (Delta_shear_square + Delta_antishear_square)
    )

