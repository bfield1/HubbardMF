#!/usr/bin/python3

from math import pi, sqrt
import unittest
import warnings
import os
import filecmp

import numpy as np

import hubbard.substrate.substrate
import hubbard.substrate.dftsubstrate
import hubbard.substrate.vaspband2np

class TestBaseSubstrate(unittest.TestCase):
    def test_init(self):
        sub = hubbard.substrate.substrate.BaseSubstrate()
        msg = " did not have the expected value."
        self.assertEqual(sub.nx, 1, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 1, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0]]), msg="glist"+msg)
    #
    def test_set_params(self):
        sub = hubbard.substrate.substrate.BaseSubstrate()
        msg = " did not have the expected value."
        sub.set_params(nx=1, ny=1)
        self.assertEqual(sub.nx, 1, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 1, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0]]), msg="glist"+msg)
        sub.set_params(nx=2)
        self.assertEqual(sub.nx, 2, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 2, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0],[1,0]]), msg="glist"+msg)
        sub.set_params(nx=3, ny=2)
        self.assertEqual(sub.nx, 3, msg="nx"+msg)
        self.assertEqual(sub.ny, 2, msg="ny"+msg)
        self.assertEqual(sub.ncells, 6, msg="ncells"+msg)
        self.assertEqual(len(sub.glist), 6, msg="len(glist)"+msg)
        # The order of points is not important, only that they are all present.
        glist2 = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1]]
        self.assertTrue(all([any([np.all(k == kp) for k in sub.glist]) for kp in glist2]),
                        msg="glist"+msg)
    #
    def test_get_coupling(self):
        sub = hubbard.substrate.substrate.BaseSubstrate(nx=2, ny=1)
        # glist = [[0,0], [1,0]]
        def msg(k,pos,couple):
            return "k="+str(k)+", pos="+str(pos)+", couple="+str(couple)+", get_coupling did not produce expected array."
        # k = [0,0]; positions = [[0,0]]
        k = [0,0]
        pos = [[0,0]]
        expected = -1/sqrt(2)*np.array([[1,1]])
        arr = sub.get_coupling(k, pos)
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,1))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,1))
        # Do multiple sites
        pos = np.zeros((3,2))
        arr = sub.get_coupling(k, pos)
        expected = -1/sqrt(2)*np.ones((3,2))
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,1))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,1))
        # Do unusual coupling
        couple = [0.5, 1, 2]
        arr = sub.get_coupling(k, pos, couple)
        expected = -1/sqrt(2) * np.array([[0.5,0.5], [1,1], [2,2]])
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,couple))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,couple))
        # Do non-zero pos
        pos = [[0.5, 0.5]]
        arr = sub.get_coupling(k, pos)
        expected = -1/sqrt(2) * np.array([[1, -1]])
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,1))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,1))
        # Non-zero k
        k = [0,0.5]
        arr = sub.get_coupling(k, pos)
        expected = -1/sqrt(2) * np.array([[1j, -1j]])
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,1))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,1))
        # Do a couple of positions
        pos = [[0,0], [0.5,0.5]]
        k = [0,0]
        arr = sub.get_coupling(k, pos)
        expected = -1/sqrt(2) * np.array([[1,1], [1,-1]])
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,1))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,1))
        k = [0, 0.5]
        arr = sub.get_coupling(k, pos)
        expected = -1/sqrt(2) * np.array([[1,1], [1j,-1j]])
        self.assertEqual(expected.shape, arr.shape, msg=msg(k,pos,1))
        self.assertAlmostEqual(np.abs(expected - arr).sum(), 0, msg=msg(k,pos,1))

class TestSquareSubstrate(unittest.TestCase):
    mycls = hubbard.substrate.substrate.SquareSubstrate
    def en(self,k):
        """Energy dispersion"""
        return -2 * (np.cos(2*pi*k[0]) + np.cos(2*pi*k[1]))
    #
    def test_init(self):
        sub = self.mycls()
        msg = " did not have the expected value."
        self.assertEqual(sub.nx, 1, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 1, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0]]), msg="glist"+msg)
        self.assertEqual(sub.t, 1, msg="t"+msg)
        self.assertEqual(sub.offset, 0, msg="offset"+msg)
    #
    def test_set_params(self):
        sub = self.mycls()
        msg = " did not have the expected value."
        sub.set_params(nx=1, ny=1, t=2, offset=-1)
        self.assertEqual(sub.nx, 1, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 1, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0]]), msg="glist"+msg)
        self.assertEqual(sub.t, 2, msg="t"+msg)
        self.assertEqual(sub.offset, -1, msg="offset"+msg)
        sub.set_params(nx=2)
        self.assertEqual(sub.nx, 2, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 2, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0],[1,0]]), msg="glist"+msg)
        self.assertEqual(sub.t, 2, msg="t"+msg)
        self.assertEqual(sub.offset, -1, msg="offset"+msg)
        sub.set_params(t=3, offset=-0.5)
        self.assertEqual(sub.nx, 2, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 2, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0],[1,0]]), msg="glist"+msg)
        self.assertEqual(sub.t, 3, msg="t"+msg)
        self.assertEqual(sub.offset, -0.5, msg="offset"+msg)
    #
    def test_dispersion(self):
        sub = self.mycls()
        msg = "dispersion was not the expected value."
        # A few semi-arbitrary k-points to test.
        ks = [[0,0], [0.5,1], [-1,0.2]]
        for k in ks:
            self.assertAlmostEqual(self.en(k), sub.dispersion(k), msg=msg)
        # Change the parameters.
        sub.set_params(t=2, offset=-1)
        for k in ks:
            self.assertAlmostEqual(2*self.en(k)-1, sub.dispersion(k), msg=msg)
        sub.set_params(t=1, offset=0, nx=2, ny=3)
        for k in ks:
            self.assertAlmostEqual(self.en([k[0]/2,k[1]/3]), sub.dispersion(k), msg=msg)
    #
    def test_get_matrix(self):
        sub = self.mycls()
        msg1 = "matrix was not the expected size."
        msg2 = "matrix was not the expected value."
        # A few semi-arbitrary k-points to test.
        ks = [[0,0], [0.5,1], [-1,0.2]]
        for k in ks:
            mat = sub.get_matrix(k)
            expected = np.array([[self.en(k)]])
            self.assertEqual(mat.size, expected.size, msg=msg1)
            self.assertAlmostEqual(np.abs(mat-expected).sum(), 0, msg=msg2)
        # A larger cell.
        sub.set_params(nx=2, ny=1)
        for k in ks:
            mat = sub.get_matrix(k)
            expected = np.diag([self.en([k[0]/2,k[1]]), self.en([1/2+k[0]/2,k[1]])])
            self.assertEqual(mat.size, expected.size, msg=msg1)
            self.assertAlmostEqual(np.abs(mat-expected).sum(), 0, msg=msg2)

class TestTriangleSubstrate(TestSquareSubstrate):
    # TriangleSubstrate inherits almost everything from SquareSubstrate,
    # so I can inherit the tests as well.
    mycls = hubbard.substrate.substrate.TriangleSubstrate
    def en(self,k):
        """Energy dispersion"""
        return -2 * (np.cos(2*pi*k[0]) + np.cos(2*pi*k[1]) + np.cos(2*pi*(k[1]-k[0])))

class TestDFTSubstrate(unittest.TestCase):
    ddir = "fixtures/"
    def en(self, k):
        # Manually calculated bilinear interpolation for band1.dat
        k = np.mod(k,1)
        if k[0] < 1/3:
            x = 3*k[0]
        elif k[0] > 2/3:
            x = 3*(1-k[0])
        else:
            x = 1
        if k[1] < 1/2:
            y = 2*k[1]
        else:
            y = 2*(1-k[1])
        return x * y
    #
    def test_init(self):
        sub = hubbard.substrate.dftsubstrate.DFTSubstrate()
        msg = " did not have the expected value."
        self.assertEqual(sub.nx, 1, msg="nx"+msg)
        self.assertEqual(sub.ny, 1, msg="ny"+msg)
        self.assertEqual(sub.ncells, 1, msg="ncells"+msg)
        self.assertTrue(np.all(sub.glist == [[0,0]]), msg="glist"+msg)
        self.assertEqual(sub.offset, 0, msg="offset"+msg)
    #
    def test_loadarray(self):
        data = np.array([[0, 0], [0, 1], [0, 1]], dtype=float)
        msg = " did not have expected value."
        sub = hubbard.substrate.dftsubstrate.DFTSubstrate()
        sub.loadarray(data, gamma=True)
        self.assertAlmostEqual(np.abs(data - sub.datagrid).sum(), 0,
                               msg="datagrid"+msg)
        self.assertAlmostEqual(float(sub.interpolated(0,0, grid=False)), 0,
                               msg="interpolated at (0,0)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(0.5,0.5,grid=False)), 1,
                               msg="interpolated at (0.5,0.5)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(1,1,grid=False)), 0,
                               msg="interpolated at (1,1)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(1/3,0.5,grid=False)), 1,
                               msg="interpolated at (1/3,0.5)"+msg)
    #
    def test_loadarray_notgamma(self):
        data = np.array([[0, 0], [0, 1], [0, 1]], dtype=float)
        msg = " did not have expected value."
        sub = hubbard.substrate.dftsubstrate.DFTSubstrate()
        sub.loadarray(data, gamma=False)
        self.assertAlmostEqual(np.abs(data - sub.datagrid).sum(), 0,
                               msg="datagrid"+msg)
        self.assertAlmostEqual(float(sub.interpolated(0,1/4, grid=False)), 0,
                               msg="interpolated at (0,1/4)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(0.5,3/4,grid=False)), 1,
                               msg="interpolated at (0.5,3/4)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(1,1/4,grid=False)), 0,
                               msg="interpolated at (1,1/4)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(1/3,3/4,grid=False)), 1,
                               msg="interpolated at (1/3,0.5)"+msg)
    #
    def test_loadtxt(self):
        sub = hubbard.substrate.dftsubstrate.DFTSubstrate()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # If it's raising warnings, something's gone wrong.
            sub.loadtxt(self.ddir+"band1.dat")
        data = np.array([[0, 0], [0, 1], [0, 1]], dtype=float)
        msg = " did not have expected value."
        self.assertAlmostEqual(np.abs(data - sub.datagrid).sum(), 0,
                               msg="datagrid"+msg)
        self.assertAlmostEqual(float(sub.interpolated(0,0, grid=False)), 0,
                               msg="interpolated at (0,0)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(0.5,0.5,grid=False)), 1,
                               msg="interpolated at (0.5,0.5)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(1,1,grid=False)), 0,
                               msg="interpolated at (1,1)"+msg)
        self.assertAlmostEqual(float(sub.interpolated(1/3,0.5,grid=False)), 1,
                               msg="interpolated at (1/3,0.5)"+msg)
    #
    def test_dispersion(self):
        sub = hubbard.substrate.dftsubstrate.DFTSubstrate(filename=self.ddir+"band1.dat")
        msg = "dispersion was not the expected value."
        # A few semi-arbitrary k-points to test.
        ks = [[0,0], [0.5,1], [-1,0.2]]
        for k in ks:
            self.assertAlmostEqual(self.en(k), sub.dispersion(k), msg=msg)
        # Change the parameters.
        sub.set_params(offset=-1)
        for k in ks:
            self.assertAlmostEqual(self.en(k)-1, sub.dispersion(k), msg=msg)
        sub.set_params(offset=0, nx=2, ny=3)
        for k in ks:
            self.assertAlmostEqual(self.en([k[0]/2,k[1]/3]), sub.dispersion(k), msg=msg)
    #
    def test_get_matrix(self):
        sub = hubbard.substrate.dftsubstrate.DFTSubstrate(filename=self.ddir+"band1.dat")
        msg1 = "matrix was not the expected size."
        msg2 = "matrix was not the expected value."
        # A few semi-arbitrary k-points to test.
        ks = [[0,0], [0.5,1], [-1,0.2]]
        for k in ks:
            mat = sub.get_matrix(k)
            expected = np.array([[self.en(k)]])
            self.assertEqual(mat.size, expected.size, msg=msg1)
            self.assertAlmostEqual(np.abs(mat-expected).sum(), 0, msg=msg2)
        # A larger cell.
        sub.set_params(nx=2, ny=1)
        for k in ks:
            mat = sub.get_matrix(k)
            expected = np.diag([self.en([k[0]/2,k[1]]), self.en([1/2+k[0]/2,k[1]])])
            self.assertEqual(mat.size, expected.size, msg=msg1)
            self.assertAlmostEqual(np.abs(mat-expected).sum(), 0, msg=msg2)

class Testvaspband2np(unittest.TestCase):
    ddir = "fixtures/"
    def test_read_eigenval(self):
        """Check that read_eigenval evaluates without errors."""
        eigen = hubbard.substrate.vaspband2np.load_eigenval(self.ddir+'EIGENVAL.dos',outcar=self.ddir+'OUTCAR.dos')
        nkpt = 217
        nband = 24
        self.assertEqual(eigen.bands.shape, (nkpt,nband), msg='EIGENVAL bands array not the right shape')
        self.assertEqual(eigen.kpoints.shape, (nkpt,3), msg='EIGENVAL kpoints array not the right shape')
    #
    def test_savetxt(self):
        """Check that savetxt_band gives the expected result."""
        tmpfile = self.ddir+'tmp'
        # Delete the temporary file if it exists
        try:
            os.remove(tmpfile)
        except FileNotFoundError:
            pass
        # Load PROCAR
        procar = hubbard.substrate.vaspband2np.load_procar(self.ddir+'PROCAR.dos',outcar=self.ddir+'OUTCAR.dos')
        # N.B. PROCAR and EIGENVAL have different precisions for the same data.
        hubbard.substrate.vaspband2np.savetxt_band(tmpfile, procar, 16, self.ddir+'OUTCAR.dos')
        # Check if it matches existing file.
        msg='Generated file did not match expected file. diff '+tmpfile+' and Ag_band17.dat for comparison.'
        self.assertTrue(filecmp.cmp(tmpfile, self.ddir+'Ag_band17.dat', shallow=False), msg=msg)
        # If we get to the end in one piece, remove the tmpfile
        os.remove(tmpfile)



if __name__ == "__main__":
    unittest.main()
