"""
isobank.utils
=============

Utility functions: PSD loading, boundary construction and metric projection
helpers.

Adapted from `mbank <https://github.com/stefanoschmidt1995/mbank>`_
(Stefano Schmidt, GPLv3), trimmed to the subset needed by isobank.
"""

import numpy as np
import warnings
from itertools import combinations

from .handlers import variable_handler

####################################################################################################################

def avg_dist(avg_match, D):
	"""
	The average distance between templates such that an injection have an average match with a cubic lattice of templates. This sets the spacing between templates.

	Parameters
	----------
		avg_match: float
			Minimum match
		D: int
			Dimensionality of the space

	Returns
	-------
		avg_dist: float
			Average distance between templates
	"""
	return 2*np.sqrt((1-avg_match)/D) #Owen

####################################################################################################################

def get_boundaries_from_ranges(variable_format, M_range, q_range,
	s1_range = (-0.99,0.99), s2_range = (-0.99,0.99), chi_range = (-0.99,0.99), theta_range = (-np.pi, np.pi), phi_range = (-np.pi/2., np.pi/2.),
	iota_range = (0, np.pi), ref_phase_range = (-np.pi, np.pi), e_range = (0., 0.5), meanano_range = (0.1, 1.)):
	"""
	Given the ranges of each quantity, it combines them in a boundary array.
	No checks are performed whether the given ranges make sense.

	Parameters
	----------
		variable_format: str
			A string to specify the variable format.
			See :class:`isobank.handlers.variable_handler` for more information

		M_range, q_range, s1_range, s2_range, chi_range, theta_range, phi_range, iota_range, ref_phase_range, e_range, meanano_range: tuple
			Ranges for each physical quantity. They will be used whenever required by the ``variable_format``
			If ``mchirpeta`` mass format is set, ``M_range`` and ``q_range`` are interpreted as mchirp and eta respectively.
			If ``logMq`` mass format is set, ``M_range`` is still interpreted as the mass and **not** the log mass.

	Returns
	-------
		boundaries: :class:`~numpy:numpy.ndarray`
			shape (2,D) -
			An array with the boundaries.
	"""
	format_info = variable_handler().format_info[variable_format]

	######
	#	Setting boundaries: shape (2,D)
	######
	if format_info['spin_format'].find('1x') >-1 and s1_range[0]<0.:
		s1_range = (0, s1_range[1])
	if format_info['spin_format'] == 'fullspins':
		if s1_range[0]< 0: s1_range = (0, s1_range[1])
		if s2_range[0]< 0: s2_range = (0, s2_range[1])

	if format_info['mass_format'] == 'logMq':
		M_range = np.log10(np.asarray(M_range))
	if format_info['mass_format'] == 'logm1logm2':
		M_range = np.log10(np.asarray(M_range))
		q_range = np.log10(np.asarray(q_range))

		#setting spin boundaries
	if format_info['spin_format'] == 'nonspinning':
		boundaries = np.array([[M_range[0], q_range[0]],[M_range[1], q_range[1]]])
	elif format_info['spin_format'] == 'chi':
		boundaries = np.array([[M_range[0], q_range[0], chi_range[0]],[M_range[1], q_range[1], chi_range[1]]])
	elif format_info['spin_format'] == 's1z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0]],[M_range[1], q_range[1], s1_range[1]]])
	elif format_info['spin_format'] == 's1z_s2z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], s2_range[0]],[M_range[1], q_range[1], s1_range[1], s2_range[1]]])
	elif format_info['spin_format'] == 's1xz':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1]]])
	elif format_info['spin_format'] == 's1xyz':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], phi_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1], phi_range[1]]])
	elif format_info['spin_format'] == 's1xz_s2z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], s2_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1], s2_range[1]]])
	elif format_info['spin_format'] == 's1xyz_s2z':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], phi_range[0], s2_range[0]],[M_range[1], q_range[1], s1_range[1], theta_range[1], phi_range[1], s2_range[1]]])
	elif format_info['spin_format'] == 'fullspins':
		boundaries = np.array([[M_range[0], q_range[0], s1_range[0], theta_range[0], phi_range[0], s2_range[0], theta_range[0], phi_range[0],],[M_range[1], q_range[1], s1_range[1], theta_range[1], phi_range[1], s2_range[1], theta_range[1], phi_range[1]]])
	else:
		raise RuntimeError("Boundaries current not implemented for the required format of spins {}: apologies :(".format(format_info['spin_format']))

	if format_info['e']:
		boundaries = np.concatenate([boundaries, [[e_range[0]], [e_range[1]]]], axis =1)
	if format_info['meanano']:
		boundaries = np.concatenate([boundaries, [[meanano_range[0]], [meanano_range[1]]]], axis =1)
	if format_info['iota']:
		boundaries = np.concatenate([boundaries, [[iota_range[0]], [iota_range[1]]]], axis =1)
	if format_info['phi']:
		boundaries = np.concatenate([boundaries, [[ref_phase_range[0]], [ref_phase_range[1]]]], axis =1)

	return boundaries #(2,D)

####################################################################################################################

def load_PSD(filename, asd = False, ifo = 'H1', df = None):
	"""
	Loads a PSD from file and returns a grid of frequency and PSD values

	Parameters
	----------
		filename: str
			Name of the file to load the PSD from (can be a txt file or an xml file)

		asd: bool
			Whether the file contains an ASD rather than a PSD

		ifo: str
			Interferometer which the PSD refers to. Only for loading a PSD from xml

		df: float
			Spacing for the grid on which the PSD is evaluated. It controls the resolution of the PSD (and also the speed of metric computation)
			If `None` is given, the grid of the PSD won't be changed

	Returns
	-------
		f: :class:`~numpy:numpy.ndarray`
			Frequency grid

		PSD: :class:`~numpy:numpy.ndarray`
			PSD evaluated on the frequency grid
	"""

	if filename.endswith('xml') or filename.endswith('xml.gz'):
			#xml loading needs the (optional) ligo.lw infrastructure
		import lal.series
		from ligo.lw.utils import load_filename
		PSD_fseries = lal.series.read_psd_xmldoc(
				load_filename(filename, verbose=False,
				contenthandler=lal.series.PSDContentHandler)
			)
		try:
			PSD_fseries = PSD_fseries[ifo]
		except KeyError:
			raise ValueError("The given PSD file doesn't have an entry for the chosen interferometer {}".format(ifo))
		f = np.linspace(PSD_fseries.f0, PSD_fseries.deltaF*PSD_fseries.data.length, PSD_fseries.data.length)
		PSD = PSD_fseries.data.data
	else:
		f, PSD = np.loadtxt(filename)[:,:2].T

	if asd: PSD = np.square(PSD)

	if df:
		new_f = np.arange(f[0], f[-1]-f[0], df)
		PSD = np.interp(new_f, f, PSD)
		f = new_f

	return f, PSD

####################################################################################################################

def project_metric(metric, axes):
	"""
	Projects the metric on the given axes.
	It follows a discussion on `stackexchange <https://math.stackexchange.com/questions/2431159/how-to-obtain-the-equation-of-the-projection-shadow-of-an-ellipsoid-into-2d-plan>`_

	Parameters
	----------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			A D dimensional metric

		axes: list, int
			The D' axes to project the metric over

	Returns
	-------
		projected_metric: :class:`~numpy:numpy.ndarray`
			shape: (D',D') -
			The projected dimensional metric

	"""
	if isinstance(axes, int): axes = [axes]
	if len(axes) == metric.shape[0]: return metric

	other_axes = [ d for d in range(metric.shape[0]) if d not in axes]

	J = metric[axes,:][:,axes] #(D',D')
	K = metric[other_axes,:][:,other_axes] #(K,K)
	L = metric[axes,:][:,other_axes] #(D',K)

	K_inv = np.linalg.inv(K)

	proj_metric = J - np.einsum('ij,jk,kl->il', L, K_inv, L.T)

	return proj_metric

def get_projector(*directions):
	"""
	Given a set of orthogonal directions it computes the projector matrix :math:`P` on the orthogonal space.
	This means that for each direction :math:`d` in directions and each point :math:`x` of the space:

	.. math::

		<d, Px> = 0

	Parameters
	----------
		directions: list
			List of the orthogonal directions ``d`` that defines the projector. Each direction has dimension D.

	Returns
	-------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			The projector (a D dimensional metric)

	"""
		#Doing some checks on the input
	directions = list(directions)
	for i, d in enumerate(directions):
		if not np.allclose(np.linalg.norm(d), 1): directions[i] = d/np.linalg.norm(d)
	for n1, n2 in combinations(directions,2):
		assert np.allclose(np.vdot(n1, n2), 0.), "Directions must be orthogonal"

		#Building the projector
	P = np.eye(len(directions[0]))

	for d in directions:
		P = P - np.outer(d,d)

	return P

def get_projected_metric(metric, min_eig = 1e-4):
	"""
	Given a metric :math:`M`, it applies the projector operator :math:`P` along the eigen-direction with eigenvalue smaller than ``min_eig``.

	The resulting metric is obtained as:

	.. math::

		M' = PMP

	Parameters
	----------
		metric: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			A D dimensional metric

		min_eig: float
			Minimum tolerable eigenvalue for ``metric``. The metric will be projected on the directions of smaller eigenvalues

	Returns
	-------
		metric_projected: :class:`~numpy:numpy.ndarray`
			shape: (D,D) -
			The D dimensional metric projected along the large eigenvalues
	"""

	metric = np.asarray(metric)
	eigval, eigvec = np.linalg.eig(metric)

	ids_ = np.where(eigval<=min_eig)[0]

	if len(ids_)==0: return metric

	P = get_projector(*eigvec[:,ids_].T)

	new_metric = np.linalg.multi_dot([P, metric,P])

	return new_metric

####################################################################################################################

def plawspace(start, stop, exp, N_points):
	"""
	Generates a grid which is `power law distributed`. It has almost the same behaviour as :func:`~numpy:numpy.logspace`.
	Helpful for a nice grid spacing in the mass sector.

	Parameters
	----------
		start: float
			Starting point of the grid

		stop: float
			End point of the grid

		exp: float
			Exponent for the equally space points

		N_points: int
			Number of points to include in the grid
	"""
	assert exp != 0, "Exponent of plawspace cannot be negative!"
	f_start = np.power(start, exp)
	f_stop = np.power(stop, exp)
	points = np.linspace(f_start, f_stop, N_points)
	points = np.power(points, 1/exp)

	points[0]=start
	if N_points>1: points[-1] = stop

	return points
