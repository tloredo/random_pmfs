"""
random_pmfs

A Python module providing classes implementing probability distributions
over probability mass functions (PMFs), including over histograms
corresponding to binned probability density functions (PDFs).

Created 2023-10-03 by Tom Loredo
"""

import numpy as np
from numpy import pi, arange, linspace, zeros, ones, empty
from numpy import array, asarray, ascontiguousarray
from numpy import mod, floor, cos, exp, log
from numpy import fft
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import stats


__all__ = ['BinnedPDF', 'DirichletPMFs']

# For integrating PDFs over bins, we'll use quadrature rules based on
# Chebyshev polynomials:  Clenshaw-Curtis (closed) and Fejer's 1st rule (open).
# 
# Despite being of lower order than the corresponding Gauss-Legendre rules,
# these rules perform essentially as well for most integrands; see:
#
#   "Is Gauss Quadrature Better than Clenshaw–Curtis?" (Trefethen 2008)
#   https://epubs.siam.org/doi/10.1137/060659831
#
# These rules are particularly useful for quadratures using many points.
# For binned PDFs, it's likely that relatively small rules will be used, so
# the advantage over Gauss-Legendre is probably not important.  But 
# Gauss-Legendre is an open rule; Clenshaw-Curtis provides a closed option
# for cases where the user wants to include boundary points in the rule.
# Alternatively, one could use a (closed) Gauss-Lobatto rule.
#
# These rules (possibly in composite form) are used by the MATLAB Chebfun
# system; see "Chebfun and numerical quadrature" (Hale & Trefethen 2012),
# https://doi.org/10.1007/s11425-012-4474-z.
#
# These implementations are copied from the Inference package, which includes
# wrappers simplifying broader use than is required here, and test cases.


def ccrule(order, dft=False):
    """
    Calculate nodes and weights for a Clenshaw-Curtis rule over [-1,1].
    
    Note that the order is the degree of the polynomial that is integrated
    exactly; the number of nodes is order+1.  This is a closed rule, with nodes
    at the boundaries.
    
    If dft=True, DFTs of the weights for the CC rule and Fejer's 2nd rule
    are additionally returned.
    
    The algorithm uses an FFT of size `order`; for very large `order`, setting
    `order` to a power of 2 will significantly improve performance.
    """
    npts = order + 1
    # The nodes are formally defined as x=cos(theta) for equally spaced
    # theta; reverse them so they increase in x (the rule is symmetric).
    nodes = cos(pi*arange(npts, dtype=float)/order)[::-1]
    # Be careful to use float here or later when dividing by odds!
    odds = arange(1, order, 2, dtype=float)
    l = len(odds)
    m = order-l
    v0 = zeros(npts)
    v0[:l] = 2./odds/(odds-2)
    v0[l] = 1./odds[-1]
    # v2 is the DFT of the weights for Fejer's 2nd rule.
    v2 = -v0[:-1] - v0[-1:0:-1]  # (up to penult) - (last to 2nd)
    g0 = -ones(order)
    g0[l] += order
    g0[m] += order
    # v2+g is the DFT of the weights for Clenshaw-Curtis.
    g = g0/(order**2 - 1 + mod(order,2))
    wts = empty(npts)
    wts[:-1] = fft.ifft(v2+g).real
    wts[-1] = wts[0]
    # Make nodes contiguous before returning.
    if dft:
        return ascontiguousarray(nodes), wts, v2+g, v2
    else:
        return ascontiguousarray(nodes), wts


def f1rule(order, dft=False):
    """
    Calculate nodes and weights for Fejer's 1st rule over [-1,1].
    
    Note that the order is the degree of the polynomial that is integrated
    exactly; the number of nodes is order+1.  This is an open rule, with
    no nodes at the boundaries.
    
    If dft=True, DFTs of the weights are additionally returned.

    The algorithm uses an FFT of size order+1; for very large order, choosing
    order+1 equal to a power of 2 will significantly improve performance.
    """
    npts = order+1
    # The nodes are formally defined as x=cos(theta) for equally spaced
    # theta; reverse them so they increase in x (the rule is symmetric).
    kvals = arange(npts, 0, -1, dtype=float)-0.5  # [npts-1/2, ..., 1/2]
    nodes = cos(pi*kvals/npts)
    l = npts // 2  # int divide
    m = npts-l
    K = arange(m, dtype=float)
    v0 = zeros(npts+1, dtype=complex)
    v0[:m] = 2*exp(1j*pi*K/npts)/(1-4*K**2)
    # v1 is the DFT of the weights for Fejer's 1st rule.
    v1 = v0[:-1] + v0[-1:0:-1].conj()  # (up to penult) - (last to 2nd)
    wts = fft.ifft(v1).real
    if dft:
        return nodes, wts, v1
    else:
        return nodes, wts


class PMF:
    """
    Container class for probability mass functions (PMFs).
    """

    def __init__(self, pmf, edges=None):
        """
        Store a PMF for subsequent access and summarization.  If the PMF
        corresponds to a binned PDF (a histogram), the `edges` argument
        should specify bin edges, used for deriving stepwise PDF estimates.
        """
        self.pmf = asarray(pmf)
        self.n = len(self.pmf)
        if edges is not None:
            self.edges = asarray(edges)
            self.l, self.u = edges[0], edges[-1]
            self.intvls = np.column_stack((self.edges[:-1], self.edges[1:]))
            self.centers = 0.5*(self.edges[:-1] + self.edges[1:])
            self.widths = self.edges[1:] - self.edges[:-1]
            self.pdf_steps = self.pmf/self.widths
        else:
            self.edges = None
            self.pdf_steps = None

        labels = np.arange(self.n)
        self.distn = stats.rv_discrete(name='PMF', values=(labels, self.pmf))

    def plot(self, axes=None, xlabel=None, ylabel=None, **kwds):
        """
        Plot the PMF as a stair plot if it's a binned PDF, otherwise as a
        bar plot using the category indices as labels.
        """
        if axes is None:
            fig, axes = subplots()
        if self.edges is None:
            raise NotImplementedError('Categorical PMF barplot not yet implemented!')
            axes.bars()
        else:
            axes.stairs(self.pmf, self.edges, **kwds)
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)

    def plot_pdf(self, axes=None, xlabel=None, ylabel=None, **kwds):
        """
        Plot a stepwised PDF using the PMF and bin definitions.

        This is valid only for PMFs with edges defined.
        """
        if axes is None:
            fig, axes = subplots()
        if self.edges is None:
            raise ValueError('PDF plot only possible for PMFs with edges (bins)!')
        axes.stairs(self.pmf/self.widths, self.edges, **kwds)
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)


class BinnedPDF:
    """
    Define a probability mass function (PMF) from a probability density
    function (PDF) and a binned interval.

    Note that the PMF is normalized over the specified interval, i.e., if
    the PDF extends beyond the interval, that part of the PDF is truncated.
    As a consequence, a proper PMF is defined even if the PDF is not
    normalized (e.g., if it is a point process intensity function).
    """
 
    def __init__(self, bins, intvl=None, pdf=None, qpts=10, qopen=True):
        """
        Define a builder for computing a probability mass function from a PDF
        via integration over contiguous bins.  If a PDF (function) is passed,
        the associated PMF is computed (and stored as a `pmf` array attribute). 

        Parameters
        ----------
        bins : int or sequence of floats
            The number of bins comprising the PMF (int), or a sequence of
            bin boundaries.
        intvl : 2-element sequence, optional
            The range spanned by the PMF, e.g., `(l, u)` for lower and upper
            limits `l` and `u`.  This is not required (or allowed) if `bins`
            is a sequence.
        pdf : function
            The PDF defining the PMF.
        qpts : int, default=10
            The number of quadrature points used per bin to integrate the
            PDF over each bin.
        qopen : bool, default=True
            Specifies whether to use an open or closed quadrature rule; if
            True, Fejer's first quadrature rule is used, otherwise, a
            (closed) Clenshaw-Curtis rule is used.
        """

        self.pdf = pdf
        self.qpts = qpts
        self.qopen = qopen

        # Define the bins.
        try:
            self.edges = asarray(bins[:])
            if intvl is not None:
                raise ValueError('Do not specify intvl when providing bin boundaries!')
            self.nbins = len(self.edges) - 1
            self.l, self.u = bins[0], bins[-1]
        except TypeError:
            self.nbins = bins
            self.l, self.u = intvl
            self.edges = linspace(self.l, self.u, self.nbins+1)

        self.intvls = np.column_stack((self.edges[:-1], self.edges[1:]))
        self.centers = 0.5*(self.edges[:-1] + self.edges[1:])
        self.widths = self.edges[1:] - self.edges[:-1]

        # Get a raw quadrature rule, over [-1,1].
        if qopen:
            self._nodes, self._wts = f1rule(qpts - 1)
        else:
            self._nodes, self._wts = ccrule(qpts - 1)

        # Shift and scale the raw rule to define rules for each bin.
        self.nodes, self.wts = [], []
        for i in range(self.nbins):
            hw = 0.5*self.widths[i]
            nodes = hw*self._nodes + self.centers[i]
            # For closed rules, explicitly (re)set the boundaries, to handle
            # roundoff in extreme cases.
            if not qopen:
                nodes[0] = self.intvls[i,0]
                nodes[-1] = self.intvls[i,1]
            self.nodes.append(nodes)
            self.wts.append(hw*self._wts)
        self.nodes = array(self.nodes)
        self.wts = array(self.wts)

        if self.pdf is None:
            self.pdf = None
            self.pmf = None
            self.norm = None
        else:
            self.set_pdf(pdf)

    def set_pdf(self, pdf):
        """
        Set the PDF used to create the PMF, and compute the PMF, stored as
        a `pmf` array attribute.

        Parameters
        ----------
        pdf : function
            A callable computing the (univariate) PDF.
        """
        self.pdf = pdf
        pmf = empty(self.nbins)
        for i in range(self.nbins):
            pmf[i] = sum(self.wts[i]*pdf(self.nodes[i]))
        self.norm = pmf.sum()  # itegral of PDF over all bins
        self.pmf = PMF(pmf/self.norm, self.edges)

    def plot_pmf(self, axes=None, xlabel=r'$x$', ylabel=r'PMF'):
        """
        Plot the PMF as a stair plot.
        """
        if axes is None:
            fig, axes = subplots()
        axes.stairs(self.pmf.pmf, self.edges, lw=2)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    def plot_pdfs(self, axes=None, xlabel=r'$x$', ylabel=r'PDF'):
        """
        Plot the PMF as a stepwise PDF, and the underlying PDF (normalized over
        the binned interval) computed at the quadrature points.
        """
        if axes is None:
            fig, axes = subplots()
        axes.stairs(self.pmf.pmf/self.widths, self.edges, lw=2)
        if self.qopen:
            xvals = np.hstack(self.nodes)
        else:
            # Avoid boundary point duplication for closed rules.
            xvals = [nodes[:-1] for nodes in self.nodes[:-1]]
            xvals.append(self.nodes[-1])
            xvals = np.hstack(xvals)
        axes.plot(xvals, self.pdf(xvals)/self.norm, lw=2)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    def plot_pdf(self, axes=None, xlabel=None, ylabel=None):
        """
        Plot the underlying PDF (normalized over
        the binned interval) computed at the quadrature points.
        """
        if axes is None:
            fig, axes = subplots()
        if self.qopen:
            xvals = np.hstack(self.nodes)
        else:
            # Avoid boundary point duplication for closed rules.
            xvals = [nodes[:-1] for nodes in self.nodes[:-1]]
            xvals.append(self.nodes[-1])
            xvals = np.hstack(xvals)
        axes.plot(xvals, self.pdf(xvals)/self.norm, lw=2)
        if xlabel:
            axes.set_xlabel(xlabel)
        if ylabel:
            axes.set_ylabel(ylabel)


class DirichletPMFs:
    """
    Impliment a Dirichlet distribution over probability mass functions.
    """
 
    def __init__(self, base, conc):
        """
        Define a Dirichlet distribution over a space of probability mass 
        functions(PMFs), defined via the base measure, `base` (the expectation
        value for the PMF as an array), and the concentration parameter, `conc`.

        As an aid to interpreting the parameters, note that Bayesian inference
        based on counting n_i events in M bins, adopting a uniform prior
        and a multinomial likelihood function, produces a Dirichlet posterior
        distribution where the base measure has values (n_i + 1)/(N + M)
        (for large numbers of counts, this is the fraction of counts in bin i),
        and the concentration parameter is conc = N + M, where N is the total
        number of counts.

        Parameters
        ----------
        base : array-like or BinnedPDF instance
            A sequence of probabilities or a BinnedPDF instance defining the
            base measure.
        conc : float
            The concentration parameter.
        """
        try:  # simple sequence case (categorical dist'n)
            self.base = asarray(base[:])
        except TypeError:  # BinnedPDF case
            self.binned_pdf = base
            self.edges = base.edges
            self.base = base.pmf.pmf
        self.conc = conc

        # Derived parameters for SciPy's Dirichlet:
        self.shape = conc*self.base

        self.distn = stats.dirichlet(self.shape)

    def sample(self):
        """
        Return a single PMF sampled from the Dirichlet distribution, as a PMF
        instance.

        For multiple samples, if just the PMFs as arrays are needed, use the
        `rvs` method (of the underlying SciPy Dirichlet RV object).
        """
        return PMF(self.distn.rvs()[0], self.edges)

    def __getattr__(self, name):
        # Capture method names to pass to the SciPy `dirichlet` frozen RV
        # instance.  We can't just subclass `stats.dirichlet` because it's
        # a generator function, not a class.
        if name in ('rvs', 'pdf', 'logpdf', 'alpha', 'entropy', 'mean',
            'var', 'random_state'):
            return getattr(self.distn, name)



if __name__ == '__main__':
    from scipy.special import loggamma
    from matplotlib import rc


    rcParams['figure.figsize'] = (7,5)
    rc('figure.subplot', bottom=.125, top=.95, left=0.1, right=.95)  # left=0.125
    rc('font', size=14)  # default for labels (not axis labels)
    rc('font', family='serif')  # default for labels (not axis labels)
    rc('axes', labelsize=14)
    rc('xtick.major', pad=8)
    rc('xtick', labelsize=12)
    rc('ytick.major', pad=8)
    rc('ytick', labelsize=12)
    rc('savefig', dpi=300)
    rc('axes.formatter', limits=(-4,4))

    ion()

    def gaussian(x):
        """
        An unnormalized Gaussian function with mean 5 and standard devn 1.
        """
        return exp(-0.5*(x-5)**2/1)

    # Gaussian histogram, equal-sized bins:
    g_pmf = BinnedPDF(20, (-1., 11.))
    g_pmf.set_pdf(gaussian)
    fig, axs = subplots(1, 2, figsize=(12,5))
    subplots_adjust(wspace=.25)
    g_pmf.plot_pdfs(axs[0])
    g_pmf.plot_pmf(axs[1])

    # Gaussian histogram, unequal-sized bins:
    bins = [-1., 1., 3., 4., 4.5, 5., 5.5, 6., 8., 11.]
    g_pmf = BinnedPDF(bins, qopen=False)
    g_pmf.set_pdf(gaussian)
    fig, axs = subplots(1, 2, figsize=(12,5))
    subplots_adjust(wspace=.25)
    g_pmf.plot_pdfs(axs[0])
    g_pmf.plot_pmf(axs[1])

    # Sample PMFs using an uniformly binned Gaussian base dist'n.
    nbins = 20
    g_pmf = BinnedPDF(nbins, (-1., 11.), pdf=gaussian)
    conc = nbins + nbins*100
    g_dir = DirichletPMFs(g_pmf, conc)
    fig, ax = subplots()
    means = zeros(nbins)
    nsamp = 30
    for i in range(nsamp):
        pmf = g_dir.sample()
        means += pmf.pmf
        pmf.plot(ax, alpha=0.5)
    means = means/nsamp
    plot(g_pmf.centers, means, 'ko')

    # Mixture of gammas cases:

    def gamma_ms(x, mu, sig):
        """
        The PDF for a gamma distribution over `x` with mean `mu` and standard
        deviation `sig`.
        """
        a = (mu/sig)**2  # shape parameter
        scale = sig**2/mu
        val = x/scale
        val = (a-1.)*log(val) - val - log(scale) - loggamma(a)
        return exp(val)


    def gamma_mix(x):
        """
        A mixture of three gamma distributions, producing a lumpy
        PDF over positive x.
        """
        return 0.3*gamma_ms(x, .4, .3) + .35*gamma_ms(x, .45, .15) + \
            .35*gamma_ms(x, .8, .15)


    nbins = 15
    gm_pmf = BinnedPDF(nbins, (0., 1.5), pdf=gamma_mix, qpts=15)
    fig, axs = subplots(1, 2, figsize=(12,5))
    subplots_adjust(wspace=.25)
    gm_pmf.plot_pdfs(axs[0])
    gm_pmf.plot_pmf(axs[1])

    # Set concentration by setting a typical number of objects
    # found per bin.
    conc = nbins + nbins*100
    gm_dir = DirichletPMFs(gm_pmf, conc)
    fig, ax = subplots()
    means = zeros(nbins)
    nsamp = 100
    for i in range(nsamp):
        pmf = gm_dir.sample()
        means += pmf.pmf
        pmf.plot_pdf(ax, alpha=.3)
    means = means/nsamp
    plot(gm_pmf.centers, means/gm_pmf.widths, 'ko')
    gm_pmf.plot_pdf(ax)

    # How fast?
    import timeit  # note timeit disables GC
    start_time = timeit.default_timer()
    pmfs = gm_dir.distn.rvs(10000)
    elapsed = timeit.default_timer() - start_time
    print('pmfs shape:', pmfs.shape)
    print('Time for 10000 samples (sec):', elapsed)
