"""
isobank.handlers
================

:class:`variable_handler`: takes care of the BBH parametrization.

Adapted from `mbank <https://github.com/stefanoschmidt1995/mbank>`_
(Stefano Schmidt, GPLv3), trimmed to the variable_handler class only.
"""
####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import itertools
import collections

	#ligo.lw imports for xml files: pip install python-ligo-lw

import lal 
import lalsimulation as lalsim

from tqdm import tqdm


import scipy.stats
import scipy.integrate
import scipy.spatial

#TODO: You should use a GMM with infinite dirichlet prior to interpolate the metric! It's gonna be much much faster (and effective) 
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
import torch

#############DEBUG LINE PROFILING
try:
	from line_profiler import LineProfiler

	def do_profile(follow=[]):
		def inner(func):
			def profiled_func(*args, **kwargs):
				try:
					profiler = LineProfiler()
					profiler.add_function(func)
					for f in follow:
						profiler.add_function(f)
					profiler.enable_by_count()
					return func(*args, **kwargs)
				finally:
					profiler.print_stats()
			return profiled_func
		return inner
except:
	pass

####################################################################################################################

###
class variable_handler(object):
	"""
	Class to handle a large number of variable layouts.
	The full BBH space is characterized by the following variables
	
	::
	
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi
	
	
	These are standard quantities in gravitational waves and the ligo waveform infrastructure can accept all these paramters.

	A *variable layout* is a set of variables that parametrize a subspace of such 12 dimensional BBH parameter space.
	The conversion between a the full space and the chosen subspace is made by means of a projection, where any variable not used in the paramterization is set to a default value of 0.
	For instance, ``(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi)`` can be mapped to ``(m1, m2, s1x, 0, s1z, 0, 0, s2z, iota, 0, e, 0)`` with a suitable variable format.
	The variable parametrizing the projected BBH will alway be labeled as ``theta`` and has a variable dimension ``D``, depending on the chosen variable format.
	
	The variable layout is specified by a string ``variable_format``, that can be passed to any function of this class.
	
	The general variable format can be built as any of the follwing:
	
	::
	
		MassFormat_SpinFormat
		MassFormat_SpinFormat_AnglesFormat	
		MassFormat_SpinFormat_EccentricityFormat
		MassFormat_SpinFormat_EccentricityFormat_AnglesFormat

	Valid format for the two masses are:
	
	- ``Mq``: Total mass and q
	- ``logMq``: Log10 of total mass and q
	- ``mceta``: chirp mass and eta
	- ``m1m2``: mass1 and mass2
	- ``logm1logm2``: Log 10 of mass1 and mass2
	
	Valid formats for spins are:
	
	- ``nonspinning``: no spins are considered (only two masses), D = 0
	- ``chi``: only ``chi_eff`` is considered. That means that ``s1z=s2z=chi``
	- ``s1z``: only the z spin component of most massive BH is considered, D = 3
	- ``s1z_s2z``: only the z components of the spins are considered (no precession), D = 2
	- ``s1xz``: spin components assigned to one BH in plane xz, D = 2
	- ``s1xyz``: spin components assigned to one BH, D = 3
	- ``s1xz_s2z``: the two z components are assigned as well as s1x, D = 3
	- ``s1xyz_s2z``: the two z components are assigned as well as s1x, s1y,  D = 4
	- ``fullspins``: all the 6 dimensional spin parameter is assigned,  D = 6
	
	Regarding the spins, we stick to the following conventions:

	- If a spin is aligned to the z axis the spin variable is the value of the z component of the spin (with sign): ``s1z`` or ``s2z``.
	- If a generic spin is assigned to a BH, the spin is *always** expressed in sperical coordinates ``s1``, ``theta1``, ``phi``. ``s1`` (or ``s2``) represents the spin magnitude (between 0 and 1). The angle ``theta1`` (``theta2``) corresponds to the polar angle of the spin, which controls the magnitude of in-plane spin. The angle ``phi1``(``phi2``) is the aximuthal angle (if set), which controls the mixing between x and y components.

	On top of the spins, the user can **optionally** specify a format for Angles and for the Eccentricity
	
	Valid formats for the angles are:
	
	- ``iota``: the inclination angle is included
	- ``iotaphi``: the inclination angle and reference phase are included

	Valid formats for the eccentricity are:
	
	- ``e``: to include the orbital eccentricity
	- ``emeanano``: to include the orbital eccentricity and mean periastron anomaly

	Note that some waveform approximants do not support all these paramters. Using a format that sets them to be non zero will likely fail. 
	
	For example, valid formats are: ``mceta_s1xz_s2z_e_iotaphi``, ``m1m2_nonspinning_e``, ``Mq_s1xz_s2z_iotaphi``, ``m1m2_s1z_s2z``
	"""
	#TODO: hard code here the values of epsilons?

	def __init__(self):
		"Initialization. Creates a dict of dict with all the info for each format" 
		
			#hard coding valid formats for masses, spins, eccentricity and angles
		self.m_formats = ['m1m2', 'Mq', 'logMq', 'logm1logm2', 'mceta'] #mass layouts
		self.s_formats = ['nonspinning', 'chi', 's1z', 's1z_s2z', 's1xz', 's1xyz', 's1xz_s2z', 's1xyz_s2z', 'fullspins'] #spin layouts
		self.e_formats = ['', 'e', 'emeanano'] #eccentric layouts
		self.angle_formats = ['', 'iota', 'iotaphi'] #angles layouts
		
			#hard coding dimensions for each format
		D_spins = {'nonspinning':0, 'chi':1, 's1z':1, 's1z_s2z':2, 's1xz':2, 's1xyz':3, 's1xz_s2z':3, 's1xyz_s2z':4, 'fullspins': 6} #dimension of each spin format
		D_ecc = {'':0, 'e':1, 'emeanano':2} #dimension of each eccentric format
		D_angles = {'':0, 'iota':1, 'iotaphi':2} #dimension of each angle format

			#creating info dictionaries
		self.format_info = {}
		self.valid_formats = []
		self.format_D = {}
			
		for m_, s_, e_, a_ in itertools.product(self.m_formats, self.s_formats, self.e_formats, self.angle_formats):
				#nonspinning and noneccentric formats don't sample angles...
			#if s_ == 'nonspinning' and (a_ != '' or e_ == ''): continue 
			format_to_add = ''
			for f_ in [m_, s_, e_, a_]:
				if f_ != '': format_to_add += '_{}'.format(f_)
			format_to_add = format_to_add[1:] #removing '_' in the first position
			
				#adding the format to the different dicts
			self.valid_formats.append(format_to_add)
			self.format_D[format_to_add] = 2+ D_spins[s_] + D_ecc[e_] + D_angles[a_] #dimension of the variable format
			self.format_info[format_to_add] = {'D':self.format_D[format_to_add],
				'mass_format': m_, 'spin_format': s_, 'eccentricity_format': e_, 'angle_format':a_,
				'e': (e_.find('e')>-1), 'meanano': (e_.find('meanano')>-1),
				'iota': (a_.find('iota')>-1) ,'phi': (a_.find('phi')>-1)}
		
			#Adding format BBH_components
		self.valid_formats.append('BBH_components')
		self.format_D['BBH_components'] = 12
		self.format_info['BBH_components'] = {'D':12,
				'mass_format': 'BBH_components', 'spin_format': 'BBH_components',
				'eccentricity_format': 'BBH_components', 'angle_format': 'BBH_components',
				'e': True, 'meanano':True,
				'iota': True,'phi': True}
		
		self.MAX_SPIN = 0.999 #defining the constant maximum value for the spin (used for any check that's being done)
		self.MAX_Q = 100.
		
		self.constraints = {'M':(0.,np.inf), 'logM':(-np.inf,np.inf), 'q': (1./self.MAX_Q, self.MAX_Q),
				'Mc':(0., np.inf), 'eta':(1./self.MAX_Q, 0.25),
				'mass1':(0, np.inf), 'mass2':(0, np.inf),
				'chi': (-self.MAX_SPIN, self.MAX_SPIN), 
				's1z': (-self.MAX_SPIN, self.MAX_SPIN), 's2z': (-self.MAX_SPIN, self.MAX_SPIN),
				#FIXME: you should think VERY VERY carefully on the way you treat the spin variables.
				#At the moment, mbank cannot cover the negative s1x region in for spin format s1xz
				's1': (0., self.MAX_SPIN), 's2': (0., self.MAX_SPIN),
				'theta1':(-np.pi, np.pi), 'phi1':(-np.pi, np.pi),
				'theta2':(-np.pi, np.pi), 'phi1':(-np.pi, np.pi),
				'iota': (0.,np.pi), 'phi': (-np.inf, np.inf),
				'e': (0., 1.), 'meanano': (0, 1.)
				} #allowed ranges for each label
		
			#Adding a set of functions to extract some values from the BBH components
		self.theta_getter = {
			'mass1': lambda x: x[:,0],
			'mass2': lambda x: x[:,1],
			'logm1': lambda x: np.log10(x[:,0]),
			'logm2': lambda x: np.log10(x[:,1]),
			'M': lambda x: x[:,0] + x[:,1],
			'logM': lambda x: np.log10(x[:,0] + x[:,1]),
			'q': lambda x: np.maximum(x[:,1] / x[:,0], x[:,0] / x[:,1]),
			'Mc': lambda x: (x[:,0] + x[:,1])*np.power(np.divide(x[:,1] * x[:,0], np.square(x[:,0] + x[:,1])), 3./5.),
			'eta': lambda x: np.divide(x[:,1] * x[:,0], np.square(x[:,0] + x[:,1]) ),
			'chi': lambda x: (x[:,0]*x[:,4] + x[:,7]*x[:,1])/(x[:,0]+x[:,1]),
			's1z': lambda x: x[:,4],
			's2z': lambda x: x[:,7],
			's1': lambda x: np.linalg.norm(x[:,2:5], axis =1),
			'theta1': lambda x: np.arctan2(np.linalg.norm(x[:,[2,3]], axis =1)*np.sign(x[:,2]), x[:,4]),
			'phi1': lambda x: np.arctan(x[:,3]/(x[:,2]+1e-20)),
			's2': lambda x: np.linalg.norm(x[:, 5:8], axis =1),
			'theta2': lambda x: np.arctan2(np.linalg.norm(x[:,[5,6]], axis = -1)*np.sign(x[:,5]), x[:,7]),
			'phi2': lambda x: np.arctan(x[:,6]/(x[:,5]+1e-20)),
			'e': lambda x: x[:,8],
			'meanano': lambda x: x[:,9],
			'iota': lambda x: x[:,10],
			'phi': lambda x: x[:,11]
		}
		for v, tg in self.theta_getter.items():
			setattr(self, 'get_'+v, tg)
		
		self.comp_getter = [
		'm1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'e', 'meanano', 'iota', 'phi'
		] #does this obj make sense?
				
		
		return

	def is_theta_ok(self, theta, variable_format, raise_error = False):
		"""
		Given a value of theta, it checks whether it is an acceptable value for the given spin format.
		It calls :func:`get_BBH_components` internally.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format
		
		variable_format: str
			How to handle the BBH variables.
		
		raise_error: bool
			Whether to raise a ValueError if theta is not acceptable
		
		Returns
		-------
			is_ok: bool
				`True` if `theta` is an acceptable value. `False` otherwise
		"""
		labels = self.labels(variable_format, latex = False)
		theta = np.atleast_2d(theta)
		
		is_ok = True
		if raise_error: bad_labels = []
		
		for i, l in enumerate(labels):
			is_ok_l = np.logical_and( self.constraints[l][0]<theta[...,i], theta[...,i]<self.constraints[l][1])
			is_ok = np.logical_and(is_ok, is_ok_l)
			if not np.all(is_ok_l) and raise_error: bad_labels.append(l)
		
		if raise_error and not np.all(is_ok):
			raise ValueError("The given theta does not have an acceptable value for the quantities: {}".format(*bad_labels))
		
		return is_ok


	def switch_BBH(self, theta, variable_format):
		"""
		Given theta, it returns the theta components of the system with switched BBH masses (so that m1>m2)
		If only BH1 has an in-plane component, only the z components of the spins will be switched: this is equivalent to assume that the in-plane spin of BH1 is a collective spin.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format
		
		variable_format: str
			How to handle the BBH variables.
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if variable_format == 'BBH_components':
			ids = np.where(theta[:,0]<theta[:,1])[0]
			theta[ids,2:5], theta[ids,5:8] =  theta[ids,5:8], theta[ids,2:5]
			return theta
		
		if self.format_info[variable_format]['mass_format'] in ['m1m2', 'logm1logm2']:
			ids = np.where(theta[:,0]<theta[:,1])[0]
			theta[ids,0], theta[ids,1] = theta[ids,1], theta[ids,0] #switching masses
		elif self.format_info[variable_format]['mass_format'] in ['Mq', 'logMq']:
			ids = np.where(theta[:,1]<1)[0]
			theta[ids,1] = 1./theta[ids,1] #switching masses
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			return theta #this mass configuration is symmetric, no further action is required
		
		if len(ids)<1: return theta
		
		if self.format_info[variable_format]['spin_format'] in ['nonspinning', 'chi','s1z']:
			pass #In chi and nonspinning there's nothing to switch #s1z is always on the larger spin
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			theta[ids,2], theta[ids,3] = theta[ids,3], theta[ids,2] #switching spins
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			pass #chiP is always intended to be on the largest BH (pay attention to this)
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			pass #chiP is always intended to be on the largest BH (pay attention to this)
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			theta[ids,3], theta[ids,4] = theta[ids,4], theta[ids,3] #switching spins
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			theta[ids,4], theta[ids,5] = theta[ids,5], theta[ids,4] #switching spins
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			theta[ids,2:5], theta[ids,5:8] =  theta[ids,5:8], theta[ids,2:5] #switching spins


		if squeeze: theta = np.squeeze(theta)
		return theta
	
	def labels(self, variable_format, latex = False):
		"""
		List the names of the variables for each entry of the BBH parameter vector
		
		Parameters
		----------
		
		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
		
		labels: list
			List of labels for the parmams in the BBH (each a str)
		
		latex: bool
			Whether the labels should be in latex
		"""
		assert variable_format in self.valid_formats, "Wrong variable format given"
		
		if variable_format == 'BBH_components':
			if latex: labels = [r'$m_1$', r'$m_2$', r'$s_{1x}$', r'$s_{1y}$', r'$s_{1z}$', r'$s_{2x}$', r'$s_{2y}$', r'$s_{2z}$', r'$e$', r'$meanano$', r'$\iota$', r'$\varphi$']
			else:  labels = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'e', 'meanano', 'iota', 'phi']
			return labels
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			if latex: labels = [r'$m_1$', r'$m_2$']
			else: labels = ['mass1', 'mass2']
		elif self.format_info[variable_format]['mass_format'] == 'logm1logm2':
			if latex: labels = [r'$\log_{10}m_1$', r'$\log_{10}m_2$']
			else: labels = ['logm1', 'logm2']
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			if latex: labels = [r'$M$', r'$q$']
			else: labels = ['M', 'q']
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			if latex: labels = [r'$\log_{10}M$', r'$q$']
			else: labels = ['logM', 'q']
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			if latex: labels = [r'$\mathcal{M}_c$', r'$\eta$']
			else: labels = ['Mc', 'eta']
		
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
		elif self.format_info[variable_format]['spin_format'] == 'chi':
			if latex: labels.extend([r'$\chi$'])
			else: labels.extend(['chi'])
		elif self.format_info[variable_format]['spin_format'] == 's1z':
			if latex: labels.extend([r'$s_{1z}$'])
			else: labels.extend(['s1z'])
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			if latex: labels.extend([r'$s_{1z}$', r'$s_{2z}$'])
			else: labels.extend(['s1z', 's2z'])
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$'])
			else: labels.extend(['s1', 'theta1'])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$'])
			else: labels.extend(['s1', 'theta1', 'phi1'])
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$s_{2z}$'])
			else: labels.extend(['s1', 'theta1', 's2z'])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$', r'$s_{2z}$'])
			else: labels.extend(['s1','theta1', 'phi1', 's2z'])
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			if latex: labels.extend([r'$s_{1}$', r'$\theta_1$', r'$\phi_1$', r'$s_{2}$', r'$\theta_2$', r'$\phi_2$'])
			else: labels.extend(['s1','theta1', 'phi1', 's2', 'theta2', 'phi2'])
		
		if self.format_info[variable_format]['e'] and latex: labels.append(r'$e$')
		if self.format_info[variable_format]['e'] and not latex: labels.append('e')

		if self.format_info[variable_format]['meanano'] and latex: labels.append(r'$meanano$')
		if self.format_info[variable_format]['meanano'] and not latex: labels.append('meanano')
		
		if self.format_info[variable_format]['iota'] and latex: labels.append(r'$\iota$')
		if self.format_info[variable_format]['iota'] and not latex: labels.append('iota')

		if self.format_info[variable_format]['phi'] and latex: labels.append(r'$\phi$')
		if self.format_info[variable_format]['phi'] and not latex: labels.append('phi')
		
		return labels
	
	def D(self, variable_format):
		"""
		Returns the dimensionality of the parameter space required
		
		Parameters
		----------
		
		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
		
		D: int
			Dimensionality of the BBH parameter vector			
		"""
		assert variable_format in self.valid_formats, "Wrong variable format given"
		return self.format_info[variable_format]['D']

	def format_info(self, variable_format):
		"""
		Returns the a dict with some information about the format.
		The dict has the following entries:
		
		- ``mass_format`` : format for the masses
		- ``spin_format`` : format for the spins
		- ``eccentricity_format`` : format for the eccentricities
		- ``angle_format`` : format for the angles
		- ``D`` : dimensionality of the BBH space
		- ``e`` : whether the variables include the eccentricity e
		- ``meanano`` : whether the variables include the mean periastron anomaly meanano
		- ``iota`` : whether the variables include the inclination iota
		- ``phi`` : whether the variables include the reference phase phi
		
		Parameters
		----------
		
		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
		
		format_info: int
			Dictionary with the info for the format
		"""
		assert variable_format in self.valid_formats, "Wrong variable format given"
		return self.format_info[variable_format]

	def convert_theta(self, theta, in_format, out_format):
		"""
		Converts the input `theta` in the `in_format` to the chosen `out_format`.
		
		This is equivalent to:
		
		.. code-block:: python
		
			from mbank import variable_handler
			vh = variable_handler()
			vh.get_theta(vh.get_BBH_components(theta, in_format), out_format)
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on in_format

		in_format: str
			Variable format of the input theta

		out_format: str
			Variable format for the output theta
		
		Returns
		-------
			theta: :class:`~numpy:numpy.ndarray`
				shape: (N,D)/(D,) -
				Components of the BBH in the chosen out_format
		"""
		return self.get_theta(self.get_BBH_components(theta, in_format), out_format)

	def get_theta(self, BBH_components, variable_format):
		"""
		Given the ``BBH components``, it returns the components suitable for the bank.
		
		Parameters
		----------
		
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape: (N,12)/(12,) -
			Parameters of the BBHs.
			The columns of the array should be: `m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi`

		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
			theta: :class:`~numpy:numpy.ndarray`
				shape: (N,D)/(D,) -
				Components of the BBH in the format suitable for the bank.
				The dimensionality depends on variable_format
		"""
		BBH_components, squeeze = self._check_theta_and_format(BBH_components, variable_format)
		
		assert BBH_components.shape[1] == 12, "The number of BBH parameter is not enough. Expected 12 [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi], given {}".format(BBH_components.shape[1])
		
		if variable_format == 'BBH_components':
			theta = [*BBH_components.T]
		else:
			theta = [self.theta_getter[l](BBH_components) for l in self.labels(variable_format)]

		theta = np.column_stack(theta)
		
		if squeeze: theta = np.squeeze(theta)
		
		return theta


	def get_BBH_components(self, theta, variable_format):
		"""
		Given ``theta``, it returns the components of the BBH suitable for lal
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,12) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
		
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape: (N,12) -
			Components of the BBH.
			Columns of the array are `m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi`.
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		assert theta.shape[1]==self.D(variable_format), "The number of BBH parameter doesn't fit into the given variable format. Expected {}, given {}".format(self.D(variable_format), theta.shape[1])
		
		if variable_format == 'BBH_components':
			if squeeze: return np.squeeze(theta)
			return theta
		
			#setting the masses
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			m1, m2 = theta[:,0], theta[:,1]
		elif self.format_info[variable_format]['mass_format'] == 'logm1logm2':
			m1, m2 = 10**theta[:,0], 10**theta[:,1]
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			m1, m2 = theta[:,0]*theta[:,1]/(1+theta[:,1]), theta[:,0]/(1+theta[:,1])
			m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2) #this is to make sure that m1>m2, also if q is less than 1
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			M = 10**theta[:,0]
			m1, m2 = M*theta[:,1]/(1+theta[:,1]), M/(1+theta[:,1])
			m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2) #this is to make sure that m1>m2, also if q is less than 1
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
				#see https://github.com/gwastro/sbank/blob/7072d665622fb287b3dc16f7ef267f977251d8af/sbank/tau0tau3.py#L215
			M = theta[:,0] / np.power(theta[:,1], 3./5.)
			if not np.all(theta[:,1]<=0.25):
				raise ValueError("Values of the symmetric mass ratio should be all <= 0.25. The given array has some entries not satisfying this: {}".format(theta[:,1]))
			temp_ = np.power(1.0 - 4.0 * theta[:,1], 0.5)
			m1 = 0.5 * M * (1.0 + temp_)
			m2 = 0.5 * M * (1.0 - temp_)
		
			#allocating variables for spins
		s1x, s1y, s1z = np.zeros(m1.shape), np.zeros(m1.shape), np.zeros(m1.shape)
		s2x, s2y, s2z = np.zeros(m1.shape), np.zeros(m1.shape), np.zeros(m1.shape)

			#dealing with spins
		if self.format_info[variable_format]['spin_format'] =='nonspinning':
			pass
		elif self.format_info[variable_format]['spin_format'] == 'chi':
			s1z, s2z = theta[:,2], theta[:,2]
		elif self.format_info[variable_format]['spin_format'] == 's1z':
			s1z = theta[:,2]
		elif self.format_info[variable_format]['spin_format'] == 's1z_s2z':
			s1z, s2z = theta[:,2], theta[:,3]
		elif self.format_info[variable_format]['spin_format'] == 's1xz':
			s1x, s1z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3])
		elif self.format_info[variable_format]['spin_format'] == 's1xyz':
			s1x, s1y, s1z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3])
		elif self.format_info[variable_format]['spin_format'] == 's1xz_s2z':
			s1x, s1z, s2z = theta[:,2]*np.sin(theta[:,3]), theta[:,2]*np.cos(theta[:,3]), theta[:,4]
		elif self.format_info[variable_format]['spin_format'] == 's1xyz_s2z':
			s1x, s1y, s1z, s2z = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4]), theta[:,2]*np.cos(theta[:,3]), theta[:,5]
		elif self.format_info[variable_format]['spin_format'] == 'fullspins':
			s1x, s1y = theta[:,2]*np.sin(theta[:,3])*np.cos(theta[:,4]), theta[:,2]*np.sin(theta[:,3])*np.sin(theta[:,4])
			s1z = theta[:,2]*np.cos(theta[:,3])
			s2x, s2y = theta[:,5]*np.sin(theta[:,6])*np.cos(theta[:,7]), theta[:,5]*np.sin(theta[:,6])*np.sin(theta[:,7])
			s2z = theta[:,5]*np.cos(theta[:,6])

			#dealing with angles and eccentricity (tricky!!)
		assign_var =  [self.format_info[variable_format]['e'], self.format_info[variable_format]['meanano'],
				self.format_info[variable_format]['iota'], self.format_info[variable_format]['phi']]
		N_to_assign = sum(assign_var)
		
		vars_to_assign = []
		
		k = 0
		for av_ in assign_var:
			if av_:
				vars_to_assign.append(theta[:,-N_to_assign+k] )
				k += 1
			else:
				vars_to_assign.append(np.zeros(m1.shape))
		e, meanano, iota, phi = vars_to_assign

			#setting spins to zero, if they need to be
		def set_zero_spin(s):
			ids_ = np.where(np.abs(s)<1e-10)[0]
			if len(ids_)>0: s[ids_] = 0.
			return s
		s1x, s1y, s1z = set_zero_spin(s1x), set_zero_spin(s1y), set_zero_spin(s1z)
		s2x, s2y, s2z = set_zero_spin(s2x), set_zero_spin(s2y), set_zero_spin(s2z)
		
		BBH_comps = np.stack([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi], axis = 1)
		
		if squeeze:	BBH_comps = np.squeeze(BBH_comps)
		
		return BBH_comps
	
	def get_mchirp(self, theta, variable_format):
		"""
		Given theta, it returns the chirp mass

		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
			mchirp: :class:`~numpy:numpy.ndarray`
				Chirp mass of each BBH
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] == 'm1m2':
			mchirp = np.power(theta[:,0]*theta[:,1], 3./5.) / np.power(theta[:,0]+theta[:,1], 1./5.)
		elif self.format_info[variable_format]['mass_format'] == 'logm1logm2':
			mchirp = np.power(10**theta[:,0]*10**theta[:,1], 3./5.) / np.power(10**theta[:,0]+10**theta[:,1], 1./5.)
		elif self.format_info[variable_format]['mass_format'] == 'Mq':
			mchirp = theta[:,0] * np.power(theta[:,1]/np.square(theta[:,1]+1), 3./5.)
		elif self.format_info[variable_format]['mass_format'] == 'logMq':
			mchirp = 10**theta[:,0] * np.power(theta[:,1]/np.square(theta[:,1]+1), 3./5.)
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			mchirp = theta[:,0]

		if squeeze: mchirp = mchirp[0]
		return mchirp

	def get_massratio(self, theta, variable_format):
		"""
		Given theta, it returns the mass ratio.

		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
			q: :class:`~numpy:numpy.ndarray`
				Chirp mass of each BBH
		"""
		theta, squeeze = self._check_theta_and_format(theta, variable_format)
		
		if self.format_info[variable_format]['mass_format'] =='m1m2':
			q = np.maximum(theta[:,1]/theta[:,0], theta[:,0]/theta[:,1])
		elif self.format_info[variable_format]['mass_format'] =='logm1logm2':
			q = 10**np.maximum(theta[:,1]-theta[:,0], theta[:,0]-theta[:,1])
		elif self.format_info[variable_format]['mass_format'] in ['Mq', 'logMq']:
			q = theta[:,1]
		elif self.format_info[variable_format]['mass_format'] == 'mceta':
			q = theta[:,1] #FIXME: compute this properly! 

		if squeeze: q = q[0]
		return q
	
	def get_chiP(self, m1, m2, s1x, s1y, s1z, s2x, s2y):
		"""
		Computes the precessing spin parameter (one dimensional) as in `1408.1810 <https://arxiv.org/abs/1408.1810>`_
		Also described in `2012.02209 <https://arxiv.org/abs/2012.02209>`_
		
		Parameters
		----------
		
		m1, m2: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Masses of the two BHs
			It assumes m1>=m2

		s1x, s1y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the primary black hole
		
		s1z: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Aligned spin for the primary black hole. Used to enforce Kerr limit in the spin parameter
		
		s2x, s2y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the secondary black hole
		
		Returns
		-------
		
		chiP: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			The precessing spin parameter
		"""
		m1, m2, s1x, s1y, s1z, s2x, s2y = np.asarray(m1), np.asarray(m2), \
			np.asarray(s1x), np.asarray(s1y), np.asarray(s1z), \
			np.asarray(s2x), np.asarray(s2y)
		
		q = m1/m2
		assert np.all(q>=1), 'm1 should be greater or equal than m2' #assert q to be greater than 1
		A1 = 2+1.5/q 
		A2 = 2+1.5*q
		s1_perp = np.sqrt(s1x**2+s1y**2) * m1**2 #(N,)/()
		s2_perp = np.sqrt(s2x**2+s2y**2) * m2**2 #(N,)/()
		sp = np.maximum(s1_perp*A1, s2_perp*A2) #(N,)/()
		
		chip = sp/(A1*m1**2) #(N,)/()
		
			#enforcing kerr limit
		norm = np.sqrt(chip**2+s1z**2) #(N,)/()
		ids_ = np.where(norm >=self.MAX_SPIN)[0]
		if len(ids_)>0: chip[ids_] = np.sqrt(self.MAX_SPIN - s1z[ids_]**2 )

		return chip

	def get_chiP_2D(self, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, only_spin1 = False):
		"""
		Computes the two dimensional precessing spin parameter as in `2012.02209 <https://arxiv.org/abs/2012.02209>`_
		The approximantion assigns a two dimensional in-plane spin to one of the two BHs.
		If the option only_spin1 is set, the in-plane spin will be always assigned to the primary BH (as in (7) in `2012.02209 <https://arxiv.org/abs/2012.02209>`_).
		
		Parameters
		----------
		
		m1, m2: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Masses of the two BHs

		s1x, s1y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the primary black hole
		
		s1z: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Aligned spin for the primary black hole. Used to enforce Kerr limit in the spin parameter (if it is the case)
		
		s2x, s2y: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			In-plane spins of the secondary black hole

		s2z: :class:`~numpy:numpy.ndarray`
			shape: (N,)/() -
			Aligned spin for the secondary black hole. Used to enforce Kerr limit in the spin parameter (if it is the case)
		
		only_spin1: bool
			Whether to assign the precessing spin only always to the primary BH.
			The default is False, as 2012.02209 suggests.
		
		Returns
		-------
		
		chiP_2D_1: :class:`~numpy:numpy.ndarray`
			shape: (N,2)/(2,) -
			In-plane (x and y) components of the two dimensional precessing spin parameter on the primary BH
		
		chiP_2D_2: :class:`~numpy:numpy.ndarray`
			shape: (N,2)/(2,) -
			In-plane (x and y) components of the two dimensional precessing spin parameter on the secondary BH
		
		"""
		#TODO: check the accuracy of this function
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = np.asarray(m1), np.asarray(m2), \
			np.asarray(s1x), np.asarray(s1y), np.asarray(s1z), \
			np.asarray(s2x), np.asarray(s2y),  np.asarray(s2z)
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = np.atleast_1d(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)

		assert np.all(m1>=m2), 'm1 should be greater or equal than m2' #assert q to be greater than 1

		S1_perp = (m1**2*np.column_stack([s1x, s1y]).T).T #(N,2)
		S2_perp = (m2**2*np.column_stack([s2x, s2y]).T).T #(N,2)
		
		S_perp = S1_perp + S2_perp
		
		S1_perp_norm= np.linalg.norm(S1_perp, axis = 1) #(N,)
		S2_perp_norm= np.linalg.norm(S2_perp, axis = 1) #(N,)
		
		if only_spin1: BH1_gtr_BH2 = np.ones(S1_perp_norm.shape, dtype = bool) #only the primary BH will have the in-plane spin
		else: BH1_gtr_BH2 = S1_perp_norm >= S2_perp_norm
	
			#computing effective spin parameters
		chi_eff_1, chi_eff_2 = np.zeros(S1_perp.shape), np.zeros(S2_perp.shape) #(N,2)
		
		if np.any(BH1_gtr_BH2): chi_eff_1[BH1_gtr_BH2,:] = S_perp[BH1_gtr_BH2] / (m1[BH1_gtr_BH2]**2+ S2_perp_norm[BH1_gtr_BH2])
		if np.any(~BH1_gtr_BH2): chi_eff_2[~BH1_gtr_BH2,:] = S_perp[~BH1_gtr_BH2] / (m2[~BH1_gtr_BH2]**2+ S1_perp_norm[~BH1_gtr_BH2])
		
			#enforcing Kerr limit
		norm_1 = np.linalg.norm( np.column_stack([*chi_eff_1.T, s1z]), axis =1) #(N,)/()
		norm_2 = np.linalg.norm( np.column_stack([*chi_eff_2.T, s2z]), axis =1) #(N,)/()
		
		ids_1 = np.where(norm_1 >self.MAX_SPIN)[0]
		ids_2 = np.where(norm_2 >self.MAX_SPIN)[0]
		
			#self.MAX_SPIN is a upper bound for the spin
		if len(ids_1)>0: chi_eff_1[ids_1] = (chi_eff_1[ids_1].T * np.sqrt(self.MAX_SPIN - s1z[ids_1]**2 ) / np.linalg.norm(chi_eff_1[ids_1], axis =1)).T
		if len(ids_2)>0: chi_eff_2[ids_2] = (chi_eff_2[ids_2].T * np.sqrt(self.MAX_SPIN - s2z[ids_2]**2 ) / np.linalg.norm(chi_eff_2[ids_2], axis =1)).T
		
		return chi_eff_1, chi_eff_2
	
	def get_chiP_BBH_components(self, BBH_components, chiP_type = 'chiP'):
		"""
		Given a set of BBH components (in the output format of :func:`get_BBH_components`) it returns the components of the same BBH in the precessing spin parameter approximation.
		This implements a mapping between the 4 dimensional in-plane spin parameter (s1x, s1y, s2x, s2y) onto a smaller space. Several options are available:
		
		- ``chiP``: performs the mapping described in eq (4.1) of arxiv/1408.1810, where the only non-zero componennt is s1x
		- ``chiP_2D``: performs the mapping described in eq (10-11) of arxiv/2012.02209. It consist in assigning a two dimensional in-plane spin to the BH which exibit more precession
		- ``chiP_2D_BH1``: performs the mapping described in eq (7) of arxiv/2012.02209. The spin parameter is the same as above but it is always assigned to the primary BH
		
		Parameters
		----------
		
		BBH_components: :class:`~numpy:numpy.ndarray`
			shape: (N,12) -
			Parameters of the BBHs.
			Each row should keep: ``m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi``
		
		chiP_type: str
			A string to determine which precessing spin approximation to set
			
			- ``chiP``: one dimensional spin parameter eq (5-6) of 2012.02209
			- ``chiP_2D``: two dimensional spin parameter eq (10-11) of 2012.02209
			- ``chiP_2D_BH1``: two dimensional spin parameter, always placed on BH1, as in eq (7) of 2012.02209
		
		Returns
		-------
		
		m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano iota, phi: :class:`~numpy:numpy.ndarray`
			Components of the BBH in the std parametrization after the spin parameter mapping has been applied.
			Each has shape (N,)
		"""
		#TODO: is this name right? get_BBH_components accepts theta, this accepts BBH_components. It may be confusing...
		
		assert chiP_type in ['chiP', 'chiP_2D', 'chiP_2D_BH1'], "Wrong format for the chiP_type: it should be one among ['chiP', 'chiP_2D', 'chiP_2D_BH1'], not {}".format(chiP_type)
		
		BBH_components, squeeze = self._check_theta_and_format(BBH_components, None)
		assert BBH_components.shape[1] == 12, "The number of BBH parameter is not enough. Expected 12 [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e, meanano, iota, phi], given {}".format(BBH_components.shape[1])
		chiP_BBH_components = np.array(BBH_components) #copying into a fresh new array
		
		if chiP_type == 'chiP':
			chiP = self.get_chiP(*chiP_BBH_components[:,:7].T)
			chiP_BBH_components[:,2] = chiP
			chiP_BBH_components[:,[3,5,6]] = 0.
		else:
			chiP_1, chiP_2 = self.get_chiP_2D(*chiP_BBH_components[:,:8].T, only_spin1 = (chiP_type == 'chiP_2D_BH1') )
			chiP_BBH_components[:,2:4] = chiP_1
			chiP_BBH_components[:,5:7] = chiP_2
		
		if squeeze: chiP_BBH_components = chiP_BBH_components[0,:]
		return tuple(comp_ for comp_ in chiP_BBH_components.T)

	def _check_theta_and_format(self, theta, variable_format):
		"""
		Performs some standard checks and preprocessing to the theta vector.
		
		Parameters
		----------
		
		theta: :class:`~numpy:numpy.ndarray`
			shape: (N,D)/(D,) -
			Parameters of the BBHs. The dimensionality depends on variable_format

		variable_format: str
			How to handle the BBH variables.
		
		Returns
		-------
			theta: :class:`~numpy:numpy.ndarray`
				Theta with two dimensions (N,D)
			
			squeeze: bool
				Whether the output array shall be squeezed
		
		"""
		if isinstance(variable_format, str):
			assert variable_format in self.valid_formats, "Wrong variable format given"
		
		theta = np.asarray(theta)
		if theta.ndim == 1:
			theta = theta[None, :]
			squeeze = True
		else:
			squeeze = False
		
		return theta, squeeze
	
####################################################################################################################
