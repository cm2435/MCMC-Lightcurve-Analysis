"""
Supporting code for PH30116 - Data analysis and research methods in observational astrophysics - University of Bath
LightCurveSimulator simulates exoplanet transits
"""
import numpy
import copy
import pylab
from MyExceptions import Hell, TheDead, Hope, InputError, StupidError, Cthulhu


class LightCurve:
    def __init__(self, t=None, flux=None, fileload=False, alwaysupdate=True, timemidpoint=0, unit='Days'):
        """
        Needs to add 1D check
        :param t:
        :param flux:
        """
        if fileload:
            dat = numpy.loadtxt(fileload)
            self._t = dat[:,0]
            self._rawflux = dat[:,1]
        else:
            if numpy.shape(t) != numpy.shape(flux):
                raise StandardError("Shape of time and flux must match, but is %s and %s" % (numpy.shape(t), numpy.shape(flux)))
            self._t = t
            self._rawflux = flux
        self._flux = copy.deepcopy(self._rawflux)
        self._n = numpy.size(self._rawflux)
        self._alwaysupdate =alwaysupdate
        if timemidpoint:
            self._t -= numpy.mean(self._t)
        self._tunit = unit
        self._error = numpy.zeros_like(self._rawflux)

    def add_noise(self, sn, update=None):
        """
        add noise
        :param sn:
        :return:
        """
        if update is None:
            update = self._alwaysupdate
        newflux = self._flux + self._flux * (self._rawflux / sn) * numpy.random.standard_normal(self._n)
        if update is True:
            self._flux = newflux
            self._error = self._flux * (self._rawflux / sn)
        else:
            return newflux

    def add_outliers(self, fracoutlier, stdoutlier, update=None):
        """
        a fraction of datapoints are catastrophic outliers
        :param fracoutlier:
        :param stdoutlier:
        :return:
        """
        n_outlier = int(fracoutlier*self._n)
        locateoutliers = numpy.random.randint(0, self._n, n_outlier)
        outliernoise = numpy.random.standard_normal(n_outlier) * stdoutlier * numpy.mean(self._rawflux)
        self._flux[locateoutliers] += outliernoise
        return self._flux

    def thin_lightcurve(self, thinfactor):
        """
        Regular thinning of lightcurve, only every nth datapoint is kept
        :param thinfactor:
        :return:
        """
        if type(thinfactor) is not int:
            raise InputError()
        thint = self._t[::thinfactor]
        thinflux = self._flux[::thinfactor]
        thinerr = self._error[::thinfactor]
        return thint, thinflux, thinerr

    def random_subsample(self, keepfrac):
        """

        :param keepfrac:
        :return:
        """
        if keepfrac <=0 or keepfrac > 1:
            raise InputError('keepfrac must be between 0 and 1.')
        mask = numpy.random.choice(self._n, int(self._n * keepfrac), replace=False)
        randt = self._t[mask]
        randflux = self._flux[mask]
        randerr = self._error[mask]
        return randt, randflux, randerr

    def realistic_sampling(self, obslength=1./24, obspernight=1, missedfrac=0.5, nightfrac=0.5):
        """
        Approximatelt simulates an actual observation.
        :param obslength: length of each individual observation
        :param obspernight: number of observations per night
        :param missed: chance an observation is missed
        :param nightfrac: fraction of the day useable as nighttime, default 50%
        :return: t, flux
        """
        if self._tunit != 'Days':
            raise InputError('unit needs to be in days')
        firstsundown = numpy.random.uniform(min(self._t), min(self._t) + 1, 1) # this randomly sets the time of sundown
        n_obsnights = int((max(self._t) - min(self._t)) - (firstsundown-min(self._t))) #number of nights in the lightcurve
        obs_t = [] #creating an empty list for output
        obs_flux = []
        obs_error = []
        for i in range(n_obsnights):
            sundown = ((i+1) * (firstsundown-min(self._t))) + firstsundown
            obstime = numpy.random.uniform(sundown, sundown+nightfrac, obspernight)
            for obs in obstime:
                checkweather = numpy.random.uniform()
                if checkweather > missedfrac:
                    currmask = (self._t > obs) & (self._t < obs + obslength)
                    for t, f, e in zip(self._t[currmask], self._flux[currmask], self._error[currmask]):
                        obs_t.append(t)
                        obs_flux.append(f)
                        obs_error.append(e)
        return obs_t, obs_flux, obs_error

    def add_baseline(self, level, sn=False):
        """
        Add a baseline level to the existing flux
        :param level:
        :param sn:
        :return:
        """
        baseline = numpy.zeros_like(self._t) + level
        if sn:
            baseline += numpy.random.standard_normal(self._n)*level/sn
        self._flux += baseline
        return self._flux

    def add_trend(self, polyparam, sn=False):
        """
        Add a polynomial with given paremeters
        :param polyparam:
        :param sn:
        :return:
        """
        f_trend = numpy.poly1d(polyparam)
        trend = f_trend(self._t)
        if sn:
            trend += trend / sn * numpy.random.standard_normal(self._n)
        self._flux += trend

    def running_average(self, n):
        """

        :param n:
        :return:
        """
        raise Hope("doesn't work...... yet")
        av = numpy.convolve(self._flux, numpy.ones((n,))/n, mode='same')
        return self._t, av

    def reset(self):
        """
        resets the flux
        :return:
        """
        self._flux = copy.deepcopy(self._rawflux)

    def plotlc(self, shiftmidzero=True):
        """
        Plots the lightcurve
        :param shiftmidzero: shift the mid point of time to be zero. Boolean
        :return:
        """
        if shiftmidzero:
            shift = numpy.mean(self._t)
        else:
            shift = 0
        pylab.plot(self._t - shift, self._flux, ls='None', marker='.')
        pylab.xlabel('Time [%s]' % self._tunit)
        pylab.ylabel('Flux')

    def plotlc_error(self, shiftmidzero=True):
        """
        Plots the lightcurve
        :param shiftmidzero: shift the mid point of time to be zero. Boolean
        :return:
        """
        if shiftmidzero:
            shift = numpy.mean(self._t)
        else:
            shift = 0
        pylab.errorbar(self._t - shift, self._flux, self._error, ls='None', marker='.')
        pylab.xlabel('Time [%s]' % self._tunit)
        pylab.ylabel('Flux')

    def getdata(self, shiftmidzero=True):
        if shiftmidzero:
            shift = numpy.mean(self._t)
        else:
            shift = 0
        return(self._t - shift, self._flux, self._error)

class ShortTransit(LightCurve):
    def __init__(self, fileload='Transit.txt'):
        LightCurve.__init__(self, fileload=fileload)


class LongLightcurve(LightCurve):
    def __init__(self, fileload='Transit_Long.txt'):
        LightCurve.__init__(self, fileload=fileload)

