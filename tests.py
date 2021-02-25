#!/usr/bin/python3

import unittest
from math import pi, cos, sqrt
import os

import numpy as np

import hubbard.kpoints.base
import hubbard.kpoints.kagome
import hubbard.substrate.base
import hubbard.substrate.kagome


class TestBaseKPoints(unittest.TestCase):
    """
    Tests hubbard.kpoints.base.
    Limited because there's not much it can do on its own.
    """
    def test_aaa_init(self):
        """First, make sure init works at all."""
        hub = hubbard.kpoints.base.HubbardKPoints(1)
        msg = " not initialised to expected value."
        self.assertEqual(hub.u, 0, msg="u"+msg)
        self.assertEqual(hub.nsites, 1, msg="nsites"+msg)
        self.assertEqual(hub.mag, 0, msg="mag"+msg)
        self.assertEqual(hub.dims, 2, msg="dims"+msg)
        self.assertTrue(np.all(hub.reclat == np.array([[1,0],[0,1]])), msg="reclat"+msg)
        self.assertTrue(np.all(hub.kmesh == np.array([[0,0]])), msg="kmesh"+msg)
        self.assertEqual(hub.kpoints, 1, msg="kpoints"+msg)
        self.assertEqual(hub.nelectup, 0, msg="nelectup"+msg)
        self.assertEqual(hub.nelectdown, 0, msg="nelectdown"+msg)
        self.assertTrue(np.all(hub.nup == np.array([0])), msg="nup"+msg)
        self.assertTrue(np.all(hub.ndown == np.array([0])), msg="ndown"+msg)
        self.assertFalse(hub.allow_fractions)
    #
    def test_set_electrons_one_site(self):
        hub = hubbard.kpoints.base.HubbardKPoints(1)
        for nup in [0,1]:
            for ndown in [0,1]:
                with self.subTest(nup=nup, ndown=ndown):
                    # Test numerical setting
                    hub.set_electrons(nup=nup,ndown=ndown)
                    self.assertEqual(hub.nelectup, nup)
                    self.assertEqual(hub.nelectdown, ndown)
                    self.assertTrue(np.all(hub.nup == np.array([nup])))
                    self.assertTrue(np.all(hub.ndown == np.array([ndown])))
                    # Test list-based setting
                    hub.set_electrons(nup=[nup],ndown=[ndown])
                    self.assertEqual(hub.nelectup, nup)
                    self.assertEqual(hub.nelectdown, ndown)
                    self.assertTrue(np.all(hub.nup == np.array([nup])))
                    self.assertTrue(np.all(hub.ndown == np.array([ndown])))
    #
    def test_set_electrons_one_site_failure(self):
        """Test that set_electrons fails as expected."""
        hub = hubbard.kpoints.base.HubbardKPoints(1)
        # Set up to something initial
        hub.set_electrons(nup=1, ndown=0)
        # Attempt a failure
        with self.assertRaises(ValueError, msg="Invalid electron number did not raise expected error."):
            hub.set_electrons(nup=0, ndown=3)
        with self.assertRaises(ValueError, msg="Invalid electron number did not raise expected error."):
            hub.set_electrons(nup=3, ndown=1)
        # Check that things are alright
        self.assertEqual(hub.nelectup, 1, msg="nelectup was inadvertently changed.")
        self.assertEqual(hub.nelectdown, 0, msg="nelectdown was inadvertently changed.")
    #
    def test_set_electrons_two_site_uniform(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        hub.set_electrons(nup=1, ndown=2, method='uniform')
        msg = " did not have the expected value."
        self.assertEqual(hub.nelectup, 1, msg="nelectup"+msg)
        self.assertEqual(hub.nelectdown, 2, msg="nelectdown"+msg)
        self.assertTrue(np.all(hub.nup == np.array([0.5,0.5])), msg="nup"+msg)
        self.assertTrue(np.all(hub.ndown == np.array([1,1])), msg="ndown"+msg)
    #
    def test_set_electrons_two_site_random(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        for i in range(5):
            with self.subTest(trial=i):
                hub.set_electrons(nup=1, ndown=2, method='random')
                msg = " did not have the expected value."
                self.assertEqual(hub.nelectup, 1, msg="nelectup"+msg)
                self.assertEqual(hub.nelectdown, 2, msg="nelectdown"+msg)
                msg = " did not have the expected total."
                self.assertAlmostEqual(hub.nup.sum(), 1, msg="nup"+msg)
                self.assertAlmostEqual(hub.ndown.sum(), 2, msg="ndown"+msg)
                msg = " had values out of bounds."
                self.assertFalse(np.any(hub.nup > 1), msg="nup"+msg)
                self.assertFalse(np.any(hub.ndown > 1), msg="ndown"+msg)
    #
    def test_set_electrons_two_site_list(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        hub.set_electrons(nup=[0.5,0.5], ndown=[1,0])
        msg = " did not have the expected value."
        self.assertEqual(hub.nelectup, 1, msg="nelectup"+msg)
        self.assertEqual(hub.nelectdown, 1, msg="nelectdown"+msg)
        self.assertTrue(np.all(hub.nup == np.array([0.5,0.5])), msg="nup"+msg)
        self.assertTrue(np.all(hub.ndown == np.array([1,0])), msg="ndown"+msg)
    #
    def test_get_charge_density(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        msg = "Did not have expected charge density."
        self.assertTrue(np.all(hub.get_charge_density() == np.array([0,0])), msg=msg)
        hub.set_electrons(nup=1, ndown=2, method='uniform')
        self.assertTrue(np.all(hub.get_charge_density() == np.array([1.5,1.5])), msg=msg)
        hub.set_electrons(nup=[0.75,0.25], ndown=[0,1])
        self.assertTrue(np.all(hub.get_charge_density() == np.array([0.75,1.25])), msg=msg)
    #
    def test_get_spin_density(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        msg = "Did not have expected spin density."
        self.assertTrue(np.all(hub.get_spin_density() == np.array([0,0])), msg=msg)
        hub.set_electrons(nup=1, ndown=2, method='uniform')
        self.assertTrue(np.all(hub.get_spin_density() == np.array([-0.5,-0.5])), msg=msg)
        hub.set_electrons(nup=[0.75,0.25], ndown=[0,1])
        self.assertTrue(np.all(hub.get_spin_density() == np.array([0.75,-0.75])), msg=msg)
    #
    def test_get_magnetization(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        msg = "Did not have expected magnetization."
        self.assertEqual(hub.get_magnetization(), 0, msg=msg)
        hub.set_electrons(nup=1, ndown=2, method='uniform')
        self.assertEqual(hub.get_magnetization(), -1, msg=msg)
        hub.set_electrons(nup=[0.75,0.25], ndown=[0,1])
        self.assertEqual(hub.get_magnetization(), 0, msg=msg)
    #
    def test_get_electron_number(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        msg = "Did not have expected electron number."
        self.assertEqual(hub.get_electron_number(), 0, msg=msg)
        hub.set_electrons(nup=1, ndown=2, method='uniform')
        self.assertEqual(hub.get_electron_number(), 3, msg=msg)
        hub.set_electrons(nup=[0.75,0.25], ndown=[0,1])
        self.assertEqual(hub.get_electron_number(), 2, msg=msg)
    #
    def test_local_magnetic_moment(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        msg = "Did not have expected local magnetic moment."
        self.assertEqual(hub.local_magnetic_moment(), 0, msg=msg)
        hub.set_electrons(nup=1, ndown=2, method='uniform')
        self.assertEqual(hub.local_magnetic_moment(), 0.5**2, msg=msg)
        hub.set_electrons(nup=[0.75,0.25], ndown=[0,1])
        self.assertEqual(hub.local_magnetic_moment(), 0.75**2, msg=msg)
    #
    def test_set_kmesh(self):
        hub = hubbard.kpoints.base.HubbardKPoints(1)
        # Monkhorst
        with self.subTest(mesh=(2,2),method='monkhorst'):
            hub.set_kmesh(2,2)
            msg = " did not have the expected value."
            self.assertEqual(hub.kpoints, 4, msg="kpoints"+msg)
            # The order of k-points is unimportant.
            kmesh2 = [[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75]]
            self.assertTrue(all([any([np.all(k == kp) for k in hub.kmesh]) for kp in kmesh2]),
                            msg="kmesh"+msg)
        with self.subTest(mesh=(2,3), method='monkhorst'):
            hub.set_kmesh(2,3)
            self.assertEqual(hub.kpoints, 6, msg="kpoints"+msg)
            kmesh2 = [[0.25,0],[0.25,1/3],[0.25,2/3],[0.75,0],[0.75,1/3],[0.75,2/3]]
            self.assertTrue(all([any([np.all(k == kp) for k in hub.kmesh]) for kp in kmesh2]),
                            msg="kmesh"+msg)
        with self.subTest(mesh=(2,2), method='gamma'):
            hub.set_kmesh(2,2,method='gamma')
            self.assertEqual(hub.kpoints, 4, msg="kpoints"+msg)
            kmesh2 = [[0,0],[0.5,0],[0,0.5],[0.5,0.5]]
            self.assertTrue(all([any([np.all(k == kp) for k in hub.kmesh]) for kp in kmesh2]),
                            msg="kmesh"+msg)
        with self.subTest(mesh=(2,3), method='gamma'):
            hub.set_kmesh(2,3,method='gamma')
            self.assertEqual(hub.kpoints, 6, msg="kpoints"+msg)
            kmesh2 = [[0,0],[0,1/3],[0,2/3],[0.5,0],[0.5,1/3],[0.5,2/3]]
            self.assertTrue(all([any([np.all(k == kp) for k in hub.kmesh]) for kp in kmesh2]),
                            msg="kmesh"+msg)
        with self.subTest(name='Set with list'):
            kmesh2 = [[0,0],[0.1,0],[0.3,0.3]]
            hub.set_kmesh(kmesh2)
            self.assertEqual(hub.kpoints, 3, msg="kpoints"+msg)
            self.assertTrue(np.all(hub.kmesh == kmesh2), msg="kpoints"+msg)
    #
    def test_set_u(self):
        hub = hubbard.kpoints.base.HubbardKPoints(1, u=3)
        msg = "u did not have the expected value."
        self.assertEqual(hub.u, 3, msg=msg)
        hub.set_u(1.5)
        self.assertEqual(hub.u, 1.5, msg=msg)
    #
    def test_set_mag(self):
        hub = hubbard.kpoints.base.HubbardKPoints(1)
        msg = "mag did not have the expected value."
        hub.set_mag(12)
        self.assertEqual(hub.mag, 12, msg=msg)
    #
    def test_potential(self):
        hub = hubbard.kpoints.base.HubbardKPoints(2)
        msg = " did not have the expected value." 
        with self.subTest(u=0, nup=[0,0], ndown=[0,0], mag=0):
            potup, potdown, offset = hub._potential()
            self.assertEqual(offset, 0, msg="offset"+msg)
            self.assertTrue(np.all(potup == np.zeros((2,2))), msg="potup"+msg)
            self.assertTrue(np.all(potdown == np.zeros((2,2))), msg="potdown"+msg)
        with self.subTest(u=0, nup=[0,0], ndown=[0,0], mag=1):
            hub.set_mag(1)
            potup, potdown, offset = hub._potential()
            self.assertEqual(offset, 0, msg="offset"+msg)
            self.assertTrue(np.all(potup == -np.eye(2)), msg="potup"+msg)
            self.assertTrue(np.all(potdown == np.eye(2)), msg="potdown"+msg)
        with self.subTest(u=1, nup=[0,0], ndown=[0,0], mag=0):
            hub.set_mag(0)
            hub.set_u(1)
            potup, potdown, offset = hub._potential()
            self.assertEqual(offset, 0, msg="offset"+msg)
            self.assertTrue(np.all(potup == np.zeros((2,2))), msg="potup"+msg)
            self.assertTrue(np.all(potdown == np.zeros((2,2))), msg="potdown"+msg)
        with self.subTest(u=4, nup=[0.25,0.75], ndown=[0,0], mag=0):
            hub.set_u(4)
            hub.set_electrons(nup=[0.25,0.75])
            potup, potdown, offset = hub._potential()
            self.assertEqual(offset, 0, msg="offset"+msg)
            self.assertTrue(np.all(potup == np.zeros((2,2))), msg="potup"+msg)
            self.assertTrue(np.all(potdown == [[1,0],[0,3]]), msg="potdown"+msg)
        with self.subTest(u=4, nup=[0.25,0.75], ndown=[1,0], mag=0):
            hub.set_electrons(ndown=[1,0])
            potup, potdown, offset = hub._potential()
            self.assertEqual(offset, -1, msg="offset"+msg)
            self.assertTrue(np.all(potup == [[4,0],[0,0]]), msg="potup"+msg)
            self.assertTrue(np.all(potdown == [[1,0],[0,3]]), msg="potdown"+msg)
    #
    def test_chemical_potential_from_states(self):
        hub = hubbard.kpoints.base.HubbardKPoints(1)
        msg = "chemical potential did not have the expected value."
        en = [-1, 1, 2, 5]
        with self.subTest(T=0, N=2):
            self.assertEqual(hub._chemical_potential_from_states(0, 2, en), 1, msg=msg)
        with self.subTest(T=0, N=4):
            self.assertEqual(hub._chemical_potential_from_states(0, 4, en), 5, msg=msg)
        with self.subTest(T=0, N=0):
            # Not properly defined what mu should be with N=0,
            # but it should be less than or equal to the bottom state.
            self.assertLessEqual(hub._chemical_potential_from_states(0, 0, en), -1, msg=msg)
        with self.subTest(T=1, N=2):
            self.assertAlmostEqual(hub._chemical_potential_from_states(1, 2, en), 1.5820802069520141, msg=msg)
        with self.subTest(T=0.5, N=2.5):
            self.assertAlmostEqual(hub._chemical_potential_from_states(0.5, 2.5, en), 2.172994535823183, msg=msg)
        en = [0, 1, 2, 3]
        with self.subTest(T=2, N=2):
            self.assertAlmostEqual(hub._chemical_potential_from_states(2, 2, en), 1.5, msg=msg)
        with self.subTest(T=0.01, N=2.5):
            self.assertAlmostEqual(hub._chemical_potential_from_states(0.01, 2.5, en), 2, msg=msg)
        with self.subTest(T=1, N=0):
            self.assertAlmostEqual(hub._chemical_potential_from_states(1, 0, en), -29, msg=msg)
    #
    def test_energy_from_states(self):
        hub = hubbard.kpoints.base.HubbardKPoints(3)
        msg = "energy not expected value."
        eup = [0, 1, 1]
        edown = [-0.5, 1, 2]
        with self.subTest(name="No electrons, 1 kpoint"):
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 0, msg=msg)
            en = hub._energy_from_states(eup, edown, 3)
            self.assertEqual(en, 3, msg=msg)
        with self.subTest(name="Some electrons, 1 kpoint"):
            hub.set_electrons(nup=1, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 0.5, msg=msg)
            hub.set_electrons(nup=2, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 1.5, msg=msg)
        with self.subTest(name="Some electrons, 2 kpoints"):
            hub.set_kmesh(2,1)
            eup = [0, 0, 1, 1, 1, 1]
            edown = [-0.5, -0.5, 1, 1, 2, 2]
            hub.set_electrons(nup=1, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 0.5, msg=msg)
            hub.set_electrons(nup=2, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 1.5, msg=msg)
    #
    def test_energy_from_states_fractional(self):
        hub = hubbard.kpoints.base.HubbardKPoints(3, allow_fractions=True)
        msg = "energy not expected value."
        eup = [0, 1, 1]
        edown = [-0.5, 1, 2]
        with self.subTest(name="No electrons, 1 kpoint"):
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 0, msg=msg)
            en = hub._energy_from_states(eup, edown, 3)
            self.assertEqual(en, 3, msg=msg)
        with self.subTest(name="Some electrons, 1 kpoint"):
            hub.set_electrons(nup=1, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 0.5, msg=msg)
            hub.set_electrons(nup=1.5, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 1, msg=msg)
            hub.set_electrons(nup=1.5, ndown=2.5)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 2, msg=msg)
        with self.subTest(name="Some electrons, 2 kpoints"):
            hub.set_kmesh(2,1)
            eup = [0, 0, 1, 1, 1, 1]
            edown = [-0.5, -0.5, 1, 1, 2, 2]
            hub.set_electrons(nup=1, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 0.5, msg=msg)
            hub.set_electrons(nup=1.5, ndown=2)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 1, msg=msg)
            hub.set_electrons(nup=1.5, ndown=2.5)
            en = hub._energy_from_states(eup, edown, 0)
            self.assertEqual(en, 2, msg=msg)
    #
    def test_energy_from_states_mu(self):
        hub = hubbard.kpoints.base.HubbardKPoints(3)
        msg = "energy not expected value."
        with self.subTest(kpoints=1):
            eup = [0, 2, 4, 6]
            edown = [1, 3, 5, 7]
            en = hub._energy_from_states(eup, edown, 0, T=0.01, mu=3)
            self.assertAlmostEqual(en, 0+1+2+3/2, msg=msg)
            en = hub._energy_from_states(eup, edown, 0, T=0.01, mu=3.5)
            self.assertAlmostEqual(en, 0+1+2+3, msg=msg)
            en = hub._energy_from_states(eup, edown, -2, T=0.01, mu=3.5)
            self.assertAlmostEqual(en, 0+1+2+3-2, msg=msg)
        with self.subTest(kpoints=2):
            hub.set_kmesh(2,1)
            eup = [0, 2, 4, 6]*2
            edown = [1, 3, 5, 7]*2
            en = hub._energy_from_states(eup, edown, 0, T=0.01, mu=3)
            self.assertAlmostEqual(en, 0+1+2+3/2, msg=msg)
            en = hub._energy_from_states(eup, edown, 0, T=0.01, mu=3.5)
            self.assertAlmostEqual(en, 0+1+2+3, msg=msg)
    #
    def test_toggle_allow_fractions(self):
        msg = "Toggling allow_fractions didn't work."
        hub = hubbard.kpoints.base.HubbardKPoints(1, allow_fractions=False)
        self.assertFalse(hub.allow_fractions,
                         msg="Unexpected initial value for allow_fractions")
        hub.toggle_allow_fractions()
        self.assertTrue(hub.allow_fractions, msg=msg)
        hub.toggle_allow_fractions()
        self.assertFalse(hub.allow_fractions, msg=msg)
    #
    def test_toggle_allow_fractions_integer_density(self):
        nup = [0.25, 0.75]
        ndown = [1, 0]
        msg = "Electron density has been inadvertently changed by toggling allow_fractions."
        hub = hubbard.kpoints.base.HubbardKPoints(2, nup=nup, ndown=ndown, allow_fractions=False)
        hub.toggle_allow_fractions()
        self.assertTrue(all(hub.nup == nup), msg=msg)
        self.assertTrue(all(hub.ndown == ndown), msg=msg)
        hub.toggle_allow_fractions()
        self.assertTrue(all(hub.nup == nup), msg=msg)
        self.assertTrue(all(hub.ndown == ndown), msg=msg)
    #
    def test_toggle_allow_fractions_fractional_density(self):
        nup = [0.6, 0.6]
        ndown = [0.8, 0.9]
        hub = hubbard.kpoints.base.HubbardKPoints(2, allow_fractions=True)
        hub.set_electrons(nup=nup, ndown=ndown)
        msg = " was not set correctly."
        self.assertTrue(all(hub.nup == nup), msg="nup"+msg)
        self.assertTrue(all(hub.ndown == ndown), msg="ndown"+msg)
        self.assertAlmostEqual(hub.nelectup, 1.2, msg="nelectup"+msg)
        self.assertAlmostEqual(hub.nelectdown, 1.7, msg="nelectdown"+msg)
        hub.toggle_allow_fractions()
        msg = " did not toggle correctly."
        self.assertFalse(hub.allow_fractions, msg="allow_fractions"+msg)
        self.assertEqual(hub.nelectup, 1, msg="nelectup"+msg)
        self.assertEqual(hub.nelectdown, 2, msg="nelectdown"+msg)
        self.assertTrue(all(hub.nup == [0.5,0.5]), msg="nup"+msg)
        self.assertTrue(all(hub.ndown == [1,1]), msg="ndown"+msg)
    #
    def test_init_fractional(self):
        nup = [0.6, 0.6]
        ndown = [0.8, 0.9]
        hub = hubbard.kpoints.base.HubbardKPoints(2, allow_fractions=True, nup=nup, ndown=ndown)
        msg = " was not set correctly."
        self.assertTrue(all(hub.nup == nup), msg="nup"+msg)
        self.assertTrue(all(hub.ndown == ndown), msg="ndown"+msg)
        self.assertAlmostEqual(hub.nelectup, 1.2, msg="nelectup"+msg)
        self.assertAlmostEqual(hub.nelectdown, 1.7, msg="nelectdown"+msg)
    #
    def test_set_electrons_fractional_list(self):
        nup = [0.6, 0.6]
        ndown = [0.8, 0.9]
        hub = hubbard.kpoints.base.HubbardKPoints(2, allow_fractions=True)
        hub.set_electrons(nup=nup, ndown=ndown)
        msg = " was not set correctly."
        self.assertTrue(all(hub.nup == nup), msg="nup"+msg)
        self.assertTrue(all(hub.ndown == ndown), msg="ndown"+msg)
        self.assertAlmostEqual(hub.nelectup, 1.2, msg="nelectup"+msg)
        self.assertAlmostEqual(hub.nelectdown, 1.7, msg="nelectdown"+msg)
    #
    def test_set_electrons_fractional_uniform(self):
        nup = 0.1
        ndown = 1.3
        hub = hubbard.kpoints.base.HubbardKPoints(2, allow_fractions=True)
        hub.set_electrons(nup=nup, ndown=ndown, method='uniform')
        msg = " was not set currectly."
        self.assertEqual(hub.nelectup, nup, msg="nelectip"+msg)
        self.assertEqual(hub.nelectdown, ndown, msg="nelectdown"+msg)
        self.assertTrue(all(hub.nup == nup/2), msg="nup"+msg)
        self.assertTrue(all(hub.ndown == ndown/2), msg="ndown"+msg)
    #
    def test_set_electrons_fractional_random(self):
        nup = 0.1
        ndown = 1.3
        hub = hubbard.kpoints.base.HubbardKPoints(2, allow_fractions=True)
        hub.set_electrons(nup=nup, ndown=ndown, method='random')
        msg = " was not set currectly."
        self.assertEqual(hub.nelectup, nup, msg="nelectip"+msg)
        self.assertEqual(hub.nelectdown, ndown, msg="nelectdown"+msg)


class TestKagomeKPoints(unittest.TestCase):
    """Tests hubbard.kpoints.kagome"""
    def test_aaa_init(self):
        """Tests if initialisation works at all."""
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints()
        msg = " not initialised to expected value."
        self.assertEqual(hub.u, 0, msg="u"+msg)
        self.assertEqual(hub.nsites, 3, msg="nsites"+msg)
        self.assertEqual(hub.mag, 0, msg="mag"+msg)
        self.assertEqual(hub.dims, 2, msg="dims"+msg)
        self.assertTrue(np.all(hub.reclat == np.array([[np.sqrt(3)/2,1/2],[-np.sqrt(3)/2,1/2]])),
                        msg="reclat"+msg)
        self.assertTrue(np.all(hub.kmesh == np.array([[0,0]])), msg="kmesh"+msg)
        self.assertEqual(hub.kpoints, 1, msg="kpoints"+msg)
        self.assertEqual(hub.nelectup, 0, msg="nelectup"+msg)
        self.assertEqual(hub.nelectdown, 0, msg="nelectdown"+msg)
        self.assertTrue(np.all(hub.nup == [0,0,0]), msg="nup"+msg)
        self.assertTrue(np.all(hub.ndown == [0,0,0]), msg="ndown"+msg)
        self.assertFalse(hub.allow_fractions)
        self.assertEqual(hub.nrows, 1, msg="nrows"+msg)
        self.assertEqual(hub.ncols, 1, msg="ncols"+msg)
        self.assertEqual(hub.t, 1, msg="t"+msg)
    #
    def test_get_coordinates(self):
        # I'm too lazy at the moment to test if the output is right.
        # I'll just go by the shape for now.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=1, ncols=1)
        msg = "kagome coordinates not right shape."
        coords = hub.get_coordinates()
        self.assertEqual(coords.shape, (3, 2), msg=msg)
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=4)
        coords = hub.get_coordinates()
        self.assertEqual(coords.shape, (3*2*4, 2), msg=msg)
    #
    def test_kinetic_1cell(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints()
        msg = "kinetic energy matrix was not what was expected"
        # kin1 and kin2 are equivalent up to a gauge transformation.
        def kin1(k):
            return -2*np.array([[0, cos(pi*k[0]), cos(pi*k[1])],
                                [cos(pi*k[0]), 0, cos(pi*(k[1]-k[0]))],
                                [cos(pi*k[1]), cos(pi*(k[1]-k[0])), 0]])
        def kin2(k):
            return -np.array([[0, 1+np.exp(2j*pi*k[0]), 1+np.exp(2j*pi*k[1])],
                              [1+np.exp(-2j*pi*k[0]), 0, 1+np.exp(2j*pi*(k[1]-k[0]))],
                              [1+np.exp(-2j*pi*k[1]), 1+np.exp(-2j*pi*(k[1]-k[0])), 0]])
        for k0 in [-1, -0.5, -0.2, 0, 0.3, 0.7, 1.2]:
            for k1 in [-1, -0.3, 0, 0.5, 1, 2]:
                with self.subTest(k=[k0,k1], t=1):
                    k = [k0,k1]
                    kin = hub.get_kinetic(k)
                    self.assertTrue(np.all(kin == kin1(k)) or np.all(kin == kin2(k)), msg=msg)
        hub.set_kinetic(2)
        for k in [[0,0],[0.5,0.7]]:
            with self.subTest(k=[k0,k1], t=2):
                kin = hub.get_kinetic(k)
                self.assertTrue(np.all(kin == 2*kin1(k)) or np.all(kin == 2*kin2(k)), msg=msg)
    #
    def test_kinetic_2cell(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=1)
        msg = "kinetic energy matrix was not what was expected"
        def kin1(k):
            arr = np.array([[0, 1, 1+np.exp(2j*pi*k[1]), 0, np.exp(2j*pi*k[0]), 0],
                            [0, 0, 1, 1, 0, np.exp(2j*pi*k[1])],
                            [0, 0, 0, 0, np.exp(-2j*pi*(k[1]-k[0])), 0],
                            [0, 0, 0, 0, 1, 1+np.exp(2j*pi*k[1])],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0]])
            return -(arr + arr.conjugate().transpose())
        # First, check that k=0 gives the right result
        kin = hub.get_kinetic([0,0])
        self.assertTrue(np.all(kin == kin1([0,0])), msg=msg)
        # Once this test passes, check k-dependence.
        for k0 in [-1, -0.5, -0.2, 0, 0.3, 0.7, 1.2]:
            for k1 in [-1, -0.3, 0, 0.5, 1, 2]:
                with self.subTest(k=[k0,k1], t=1):
                    k = [k0,k1]
                    kin = hub.get_kinetic(k)
                    self.assertTrue(np.all(kin == kin1(k)), msg=msg)
        hub.set_kinetic(2)
        for k in [[0,0],[0.5,0.7]]:
            with self.subTest(k=[k0,k1], t=2):
                kin = hub.get_kinetic(k)
                self.assertTrue(np.all(kin == 2*kin1(k)), msg=msg)
    #
    def test_set_kinetic_random(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints()
        hub.set_kinetic_random(t=1, wt=1, we=1)
        kin0 = hub.get_kinetic([0,0])
        ts = np.array([kin0[0,1].real/(-2), kin0[0,2].real/(-2), kin0[1,2].real/(-2)])
        es = np.diag(kin0.real)
        self.assertTrue(all((ts >= 0.5) & (ts <= 1.5)), msg="Random t's out of range.")
        self.assertTrue(all((es >= -0.5) & (es <= 0.5)), msg="Random e's out of range.")
        def kin1(k):
            return np.array([[es[0], -ts[0]*(1+np.exp(2j*pi*k[0])), -ts[1]*(1+np.exp(2j*pi*k[1]))],
                             [-ts[0]*(1+np.exp(-2j*pi*k[0])), es[1], -ts[2]*(1+np.exp(2j*pi*(k[1]-k[0])))],
                             [-ts[1]*(1+np.exp(-2j*pi*k[1])), -ts[2]*(1+np.exp(-2j*pi*(k[1]-k[0]))), es[2]]])
        msg = "kinetic energy matrix was not what was expected"
        for k0 in [-1, -0.5, -0.2, 0, 0.3, 0.7, 1.2]:
            for k1 in [-1, -0.3, 0, 0.5, 1, 2]:
                k = [k0,k1]
                kin = hub.get_kinetic(k)
                self.assertAlmostEqual(np.sum(np.abs(kin - kin1(k))), 0, msg=msg)
    #
    def test_copy(self):
        # If two Hubbard systems are identical, they should have the same
        # electron density, eigenvalues and eigenvectors.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=3, allow_fractions=True,
                                   nup=2.5, ndown=12, u=5, t=2, method='random')
        hub.set_kmesh(5,4, method='gamma')
        hub.set_mag(1)
        hub2 = hub.copy()
        self.assertTrue(np.all(hub.nup == hub2.nup),
                        msg="nup was not properly copied.")
        self.assertTrue(np.all(hub.ndown == hub2.ndown),
                        msg="ndown was not properly copied.")
        eup, edown, vup, vdown = hub._eigensystem()
        eup2, edown2, vup2, vdown2 = hub2._eigensystem()
        msg = "do not match between the copies."
        self.assertEqual(np.abs(eup - eup2).sum(), 0, msg="eup"+msg)
        self.assertEqual(np.abs(edown - edown2).sum(), 0, msg="edown"+msg)
        self.assertEqual(np.abs(vup - vup2).sum(), 0, msg="vup"+msg)
        self.assertEqual(np.abs(vdown - vdown2).sum(), 0, msg="vdown"+msg)
    #
    def test_copy_random(self):
        # As above, but uses set_kinetic_random
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=3, allow_fractions=True,
                                   nup=2.5, ndown=12, u=5, t=2, method='random')
        hub.set_kinetic_random(2, wt=1, we=2)
        hub.set_kmesh(5,4, method='gamma')
        hub.set_mag(1)
        hub2 = hub.copy()
        self.assertTrue(np.all(hub.nup == hub2.nup),
                        msg="nup was not properly copied.")
        self.assertTrue(np.all(hub.ndown == hub2.ndown),
                        msg="ndown was not properly copied.")
        eup, edown, vup, vdown = hub._eigensystem()
        eup2, edown2, vup2, vdown2 = hub2._eigensystem()
        msg = "do not match between the copies."
        self.assertEqual(np.abs(eup - eup2).sum(), 0, msg="eup"+msg)
        self.assertEqual(np.abs(edown - edown2).sum(), 0, msg="edown"+msg)
        self.assertEqual(np.abs(vup - vup2).sum(), 0, msg="vup"+msg)
        self.assertEqual(np.abs(vdown - vdown2).sum(), 0, msg="vdown"+msg)

class TestKagomeKPointsIO(unittest.TestCase):
    """Tests file reading and writing methods of KagomeHubbardKPoints."""
    ddir = './fixtures/'
    #
    def test_load(self):
        # A pre-generated file is present.
        # I'll test if I can load it successfully and if I get results
        # previously obtained.
        msg = " was not properly copied."
        ddir = self.ddir+"kagome1"
        try:
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints.load(ddir+'.json')
        except FileNotFoundError:
            print("Testing file not found: "+ddir+".json")
            self.skipTest("Testing file not found: "+ddir+".json")
        with self.subTest("Density test"):
            try:
                nup2 = np.load(ddir+'_nup.npy')
                ndown2 = np.load(ddir+'_ndown.npy')
            except FileNotFoundError:
                print("Testing files for density not found.")
            else:
                self.assertAlmostEqual(np.abs(hub.nup - nup2).sum(), 0, msg="nup"+msg)
                self.assertAlmostEqual(np.abs(hub.ndown - ndown2).sum(), 0, msg="ndown"+msg)
        with self.subTest("Eigensystem test"):
            try:
                eup2 = np.load(ddir+'_eup.npy')
                edown2 = np.load(ddir+'_edown.npy')
                vup2 = np.load(ddir+'_vup.npy')
                vdown2 = np.load(ddir+'_vdown.npy')
            except FileNotFoundError:
                print("Testing files of eigensystem not found.")
            else:
                eup, edown, vup, vdown = hub._eigensystem()
                self.assertEqual(np.abs(eup - eup2).sum(), 0, msg="eup"+msg)
                self.assertEqual(np.abs(edown - edown2).sum(), 0, msg="edown"+msg)
                self.assertEqual(np.abs(vup - vup2).sum(), 0, msg="vup"+msg)
                self.assertEqual(np.abs(vdown - vdown2).sum(), 0, msg="vdown"+msg)
    #
    def test_save(self):
        # If two Hubbard systems are identical, they should have the same
        # electron density, eigenvalues and eigenvectors.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=3, allow_fractions=True,
                                   nup=2.5, ndown=12, u=5, t=2, method='random')
        hub.set_kmesh(5,4, method='gamma')
        hub.set_mag(1)
        fname = self.ddir+'tmp.json'
        try:
            hub.save(fname)
            hub2 = hubbard.kpoints.kagome.KagomeHubbardKPoints.load(fname)
        finally:
            os.remove(fname)
        self.assertTrue(np.all(hub.nup == hub2.nup),
                        msg="nup was not properly copied.")
        self.assertTrue(np.all(hub.ndown == hub2.ndown),
                        msg="ndown was not properly copied.")
        eup, edown, vup, vdown = hub._eigensystem()
        eup2, edown2, vup2, vdown2 = hub2._eigensystem()
        msg = "do not match between the copies."
        self.assertEqual(np.abs(eup - eup2).sum(), 0, msg="eup"+msg)
        self.assertEqual(np.abs(edown - edown2).sum(), 0, msg="edown"+msg)
        self.assertEqual(np.abs(vup - vup2).sum(), 0, msg="vup"+msg)
        self.assertEqual(np.abs(vdown - vdown2).sum(), 0, msg="vdown"+msg)
    #
    def test_save_random(self):
        # As above, but uses set_kinetic_random
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=3, allow_fractions=True,
                                   nup=2.5, ndown=12, u=5, t=2, method='random')
        hub.set_kinetic_random(2, wt=1, we=2)
        hub.set_kmesh(5,4, method='gamma')
        hub.set_mag(1)
        fname = self.ddir+'tmp.json'
        try:
            hub.save(fname)
            hub2 = hubbard.kpoints.kagome.KagomeHubbardKPoints.load(fname)
        finally:
            os.remove(fname)
        self.assertTrue(np.all(hub.nup == hub2.nup),
                        msg="nup was not properly copied.")
        self.assertTrue(np.all(hub.ndown == hub2.ndown),
                        msg="ndown was not properly copied.")
        eup, edown, vup, vdown = hub._eigensystem()
        eup2, edown2, vup2, vdown2 = hub2._eigensystem()
        msg = "do not match between the copies."
        self.assertEqual(np.abs(eup - eup2).sum(), 0, msg="eup"+msg)
        self.assertEqual(np.abs(edown - edown2).sum(), 0, msg="edown"+msg)
        self.assertEqual(np.abs(vup - vup2).sum(), 0, msg="vup"+msg)
        self.assertEqual(np.abs(vdown - vdown2).sum(), 0, msg="vdown"+msg)



class TestBaseKagomeKPoints(unittest.TestCase):
    """
    While TestKagomeKPoints tests methods unique to KagomeHubbardKPoints,
    TestBaseKagomeKPoints tests method which appear in HubbardKPoints but
    can only be tested in a child of HubbardKPoints.
    """
    # Eigenenergies of kagome TB model (U=0).
    def e1(self, k):
        return -1 - np.sqrt(4*(np.cos(pi*k[0])**2 + np.cos(pi*k[1])**2 + np.cos(pi*(k[1]-k[0]))**2) - 3)
    def e2(self, k):
        return -1 + np.sqrt(4*(np.cos(pi*k[0])**2 + np.cos(pi*k[1])**2 + np.cos(pi*(k[1]-k[0]))**2) - 3)
    def e3(self, k):
        return 2
    #
    def test_eigensystem_simplest(self):
        """
        U=0, single unit cell. Theoretical results known, at least for energies.
        """
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints()
        # First, Gamma point
        eup, edown, vup, vdown = hub._eigensystem()
        self.assertAlmostEqual(np.abs(eup - [-4, 2, 2]).sum(), 0,
                               msg="Did not get expected eigenvalues.")
        self.assertAlmostEqual(np.abs(eup - edown).sum(), 0,
                               msg="Unexpected asymmetry between up and down spins.")
        # With U=0, the spin up and down Hamiltonians should be identical.
        # If the linalg methods are deterministic, then this means their outputs
        # should be identical.
        # If it weren't for degeneracies, I could test the eigenvectors more directly.
        # But because of degeneracy, direct tests are difficult.
        self.assertAlmostEqual(np.abs(vup - vdown).sum(), 0,
                msg="Unexpected asymmetry between eigenvectors of up and down spins.")
        # Now, multiple k-points.
        # Make a k-mesh
        hub.set_kmesh(4,9)
        eup, edown, vup, vdown = hub._eigensystem()
        # Determine expected eigenvalues.
        eigen = [self.e1(k) for k in hub.kmesh]
        eigen += [self.e2(k) for k in hub.kmesh]
        eigen += [self.e3(k) for k in hub.kmesh]
        eigen = sorted(eigen)
        self.assertAlmostEqual(np.abs(eup - eigen).sum(), 0,
                               msg="Did not get expected eigenvalues.")
        self.assertAlmostEqual(np.abs(eup - edown).sum(), 0,
                               msg="Unexpected asymmetry between up and down spins.")
        self.assertAlmostEqual(np.abs(vup - vdown).sum(), 0,
                msg="Unexpected asymmetry between eigenvectors of up and down spins.")
    #
    def test_eigensystem_U0_biggercell(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=3)
        # A larger unit cell can be treated like a single unit cell but with
        # the Brillouin zone subdivided.
        hub.set_kmesh(4,9)
        kmesh = np.asarray(hub.kmesh)
        arrs = []
        for i in range(2):
            for j in range(3):
                arrs.append(kmesh/np.array([2,3]) + [i/2, j/3])
        newkmesh = np.vstack(arrs)
        eup, edown, vup, vdown = hub._eigensystem()
        # Determine expected eigenvalues.
        eigen = [self.e1(k) for k in newkmesh]
        eigen += [self.e2(k) for k in newkmesh]
        eigen += [self.e3(k) for k in newkmesh]
        eigen = sorted(eigen)
        self.assertAlmostEqual(np.abs(eup - eigen).sum(), 0,
                               msg="Did not get expected eigenvalues.")
        self.assertAlmostEqual(np.abs(eup - edown).sum(), 0,
                               msg="Unexpected asymmetry between up and down spins.")
        self.assertAlmostEqual(np.abs(vup - vdown).sum(), 0,
                msg="Unexpected asymmetry between eigenvectors of up and down spins.") 
    #
    def test_eigensystem_U_uniform(self):
        # U behaves like an on-site potential
        # With uniform electron density, eigenvalues are just shifted vertically.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=2, nup=2, ndown=1, method='uniform')
        hub.set_kmesh(3,2, method='gamma')
        eup, edown, vup, vdown = hub._eigensystem()
        # Determine expected eigenvalues.
        eigen = [self.e1(k) for k in hub.kmesh]
        eigen += [self.e2(k) for k in hub.kmesh]
        eigen += [self.e3(k) for k in hub.kmesh]
        eigen = np.asarray(sorted(eigen))
        self.assertAlmostEqual(np.abs(eigen + 2/3 - eup).sum(), 0,
                msg = "Did not get expected spin up eigenvalues.")
        self.assertAlmostEqual(np.abs(eigen + 4/3 - edown).sum(), 0,
                msg = "Did not get expected spin down eigenvalues.")
        # Due to degeneracy and broken symmetry,
        # we can't directly compare the eigenvectors.
    def test_eigensystem_t0(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(t=0, u=1, nup=0, ndown=0)
        msg = "Did not get expected eigenvalues."
        eup, edown, vup, vdown = hub._eigensystem()
        self.assertEqual(np.abs(eup - [0,0,0]).sum(), 0, msg=msg)
        self.assertEqual(np.abs(edown - [0,0,0]).sum(), 0, msg=msg)
        nup = [0.3, 0.6, 0.1]
        ndown = [0.9, 1, 0.1]
        hub.set_electrons(nup=nup, ndown=ndown)
        eup, edown, vup, vdown = hub._eigensystem()
        self.assertListEqual(eup.tolist(), sorted(ndown), msg=msg)
        self.assertListEqual(edown.tolist(), sorted(nup), msg=msg)
        # Because we have no degeneracies, and all states are localised,
        # the eigenvectors are well-known.
        vupexp = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).transpose()
        vdownexp = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).transpose()
        msg = "Did not get expected eigenvectors."
        self.assertAlmostEqual(np.abs(vup - vupexp).sum(), 0, msg=msg)
        self.assertAlmostEqual(np.abs(vdown - vdownexp).sum(), 0, msg=msg)
    #
    def test_energy(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=3)
        hub.set_kmesh(5,5, method='gamma')
        # No electrons. Energy should be zero.
        msg = "Did not get expected energy."
        self.assertAlmostEqual(hub.energy(), 0, msg=msg)
        # Add one electron. This will fill the bottom band only.
        hub.set_electrons(nup=1)
        expected = sum([self.e1(k) for k in hub.kmesh])/hub.kpoints
        self.assertAlmostEqual(hub.energy(), expected, msg=msg)
        # Have two electrons, uniform, in the bottom band.
        hub.set_electrons(nup=1, ndown=1, method='uniform')
        # Expected: two lots of the last band energy,
        # both shifted up by U*n=3*1/3 (times 2, for up and down),
        # Minus U <nup> <ndown> = 1
        expected = expected * 2 + 1
        self.assertAlmostEqual(hub.energy(), expected, msg=msg)
        # Now an arbitrary example.
        hub.set_electrons(nup=[1,0.5,0.5], ndown=[0,0.5,0.5])
        self.assertAlmostEqual(hub.energy(), -2.358189735881097, msg=msg)
    #
    def test_energy_T(self):
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=3)
        hub.set_kmesh(5,5, method='gamma')
        # No electrons. Energy should be zero.
        msg = "Did not get expected energy."
        self.assertAlmostEqual(hub.energy(T=1), 0, msg=msg)
        # Now some arbitrary examples.
        hub.set_electrons(nup=[1,0.5,0.5], ndown=[0,0.5,0.5])
        self.assertAlmostEqual(hub.energy(T=1), -1.4866311209589655, msg=msg)
        self.assertAlmostEqual(hub.energy(T=0.1), -2.5809756795204564, msg=msg)
    #
    def test_chemical_potential(self):
        # Test an arbitrary example.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=2, nrows=2, ncols=2)
        hub.set_kmesh(3,3, method='gamma')
        hub.set_electrons(nup=3, ndown=3, method='uniform')
        msg = "Did not get expected chemical potential."
        self.assertAlmostEqual(hub.chemical_potential(1), -1.4255302968938393, msg=msg)
        self.assertAlmostEqual(hub.chemical_potential(0.1), -1.4865531063827795, msg=msg)
    #
    def test_eigenstep(self):
        # First, a trivial test: U=0, so it converges to the ground state in one step.
        # Also Gamma point only, which makes the energy eigenstate predictable.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=0, nup=1, ndown=1, method='random')
        msg = " did not have the expected value."
        en, nup, ndown = hub._eigenstep()
        self.assertAlmostEqual(en, -8, msg="Energy"+msg)
        self.assertAlmostEqual(np.abs(nup - [1/3,1/3,1/3]).sum(), 0, msg="nup"+msg)
        self.assertAlmostEqual(np.abs(ndown - [1/3,1/3,1/3]).sum(), 0, msg="ndown"+msg)
        # Now a more elaborate example
        hub.set_kmesh(5,5)
        hub.set_u(10)
        hub.set_electrons(nup=[1,0,0], ndown=[0,0,1])
        en, nup, ndown = hub._eigenstep()
        n1 = 0.48808139
        n2 = 0.02383723
        self.assertAlmostEqual(en, -3.157996692682141, msg="Energy"+msg)
        self.assertAlmostEqual(np.abs(nup - [n1,n1,n2]).sum(), 0, msg="nup"+msg)
        self.assertAlmostEqual(np.abs(ndown - [n2,n1,n1]).sum(), 0, msg="ndown"+msg)
    #
    def test_eigenstep_finite_T(self):
        # First, a trivial test: U=0, so it converges to the ground state in one step.
        # Also Gamma point only, which makes the energy eigenstate predictable.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=0, nup=1, ndown=1, method='random')
        msg = " did not have the expected value."
        en, nup, ndown = hub._eigenstep_finite_T(0.1)
        self.assertAlmostEqual(en, -8, msg="Energy"+msg)
        self.assertAlmostEqual(np.abs(nup - [1/3,1/3,1/3]).sum(), 0, msg="nup"+msg)
        self.assertAlmostEqual(np.abs(ndown - [1/3,1/3,1/3]).sum(), 0, msg="ndown"+msg)
        # Now a more elaborate example
        hub.set_kmesh(5,5)
        hub.set_u(10)
        hub.set_electrons(nup=[1,0,0], ndown=[0,0,1])
        en, nup, ndown = hub._eigenstep_finite_T(0.1)
        n1 = 0.48807559
        n2 = 0.02384882
        self.assertAlmostEqual(en, -3.155852250599545, msg="Energy"+msg)
        self.assertAlmostEqual(np.abs(nup - [n1,n1,n2]).sum(), 0, msg="nup"+msg)
        self.assertAlmostEqual(np.abs(ndown - [n2,n1,n1]).sum(), 0, msg="ndown"+msg)
        en, nup, ndown = hub._eigenstep_finite_T(1)
        n1 = 0.48938719
        n2 = 0.02122561
        self.assertAlmostEqual(en, -2.206683245053485, msg="Energy"+msg)
        self.assertAlmostEqual(np.abs(nup - [n1,n1,n2]).sum(), 0, msg="nup"+msg)
        self.assertAlmostEqual(np.abs(ndown - [n2,n1,n1]).sum(), 0, msg="ndown"+msg)
    #
    def test_residual(self):
        # First, test a known case
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=0, nup=[1,0,0], ndown=[0,0,1])
        hub.set_kmesh(1,1) # Have to use Gamma point only, otherwise degeneracies cause problems.
        msg = "Residual was not the expected value."
        res = hub.residual()
        self.assertAlmostEqual(res, np.sqrt(4+1+1+1+1+4)/3, msg=msg)
        # Now a simpler case.
        hub.set_electrons(nup=1, ndown=1, method='uniform')
        res = hub.residual()
        self.assertAlmostEqual(res, 0, msg=msg)
        # Now finite U, which lifts many degeneracies.
        hub.set_electrons(nup=[1,0,0], ndown=[0,0,1])
        hub.set_kmesh(5,5)
        hub.set_u(10)
        res = hub.residual() 
        n1 = 0.48808139
        n2 = 0.02383723
        self.assertAlmostEqual(res, np.sqrt(((1-n1)**2 + n1**2 + n2**2) * 2), msg=msg)
    #
    def test_residual_T(self):
        # At finite T, degeneracies are handled properly.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=0, nup=[1,0,0], ndown=[0,0,1])
        hub.set_kmesh(5,5)
        msg = "Residual was not the expected value."
        res = hub.residual(0.1)
        self.assertAlmostEqual(res, np.sqrt(4+1+1+1+1+4)/3, msg=msg)
        # Now a simpler case.
        hub.set_electrons(nup=1, ndown=1, method='uniform')
        res = hub.residual(0.1)
        self.assertAlmostEqual(res, 0, msg=msg)
        # Now finite U
        hub.set_electrons(nup=[1,0,0], ndown=[0,0,1])
        hub.set_kmesh(5,5)
        hub.set_u(10)
        res = hub.residual(0.1)
        n1 = 0.48807559
        n2 = 0.02384882
        self.assertAlmostEqual(res, np.sqrt(((1-n1)**2 + n1**2 + n2**2) * 2), msg=msg)
        res = hub.residual(1)
        n1 = 0.48938719
        n2 = 0.02122561
        self.assertAlmostEqual(res, np.sqrt(((1-n1)**2 + n1**2 + n2**2) * 2), msg=msg)
    #
    def test_eigenvalues_at_kpoints(self):
        # We'll compare to U=0 case, since nice and simple.
        hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(u=0)
        hub.set_kmesh(3,8)
        msg = "Did not get expected list of eigenvalues."
        # Expected eigenvalues
        eigen = np.array([[[self.e1(k), self.e2(k), self.e3(k)] for s in range(2)] for k in hub.kmesh])
        with self.subTest(test="Default klist"):
            bands = hub.eigenvalues_at_kpoints()
            self.assertAlmostEqual(np.abs(bands - eigen).sum(), 0, msg=msg)
            self.assertSequenceEqual(bands.shape, eigen.shape, msg="Did not have expected shape.")
        with self.subTest(test="Non-zero mag"):
            hub.set_mag(2)
            eigen += np.array([[[-2],[2]]])
            bands = hub.eigenvalues_at_kpoints()
            self.assertAlmostEqual(np.abs(bands - eigen).sum(), 0, msg=msg)
            self.assertSequenceEqual(bands.shape, eigen.shape, msg="Did not have expected shape.")
        with self.subTest(test="Specified k-points"):
            hub.set_mag(0)
            klist = [[0,0], [0.5,0.5]]
            eigen = np.array([[[self.e1(k), self.e2(k), self.e3(k)] for s in range(2)] for k in klist])
            bands = hub.eigenvalues_at_kpoints(klist)
            self.assertAlmostEqual(np.abs(bands - eigen).sum(), 0, msg=msg)
            self.assertSequenceEqual(bands.shape, eigen.shape, msg="Did not have expected shape.")
    #
    def test_linear_mixing(self):
        msg1 = "Density did not have expected value."
        msg2 = "Number of electrons changed."
        msg3 = "Residual not below expected threshold."
        with self.subTest(test="U=0"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=0)
            hub.set_electrons(nup=4, ndown=7, method='random')
            # Gamma point only, because degeneracies. But even then, must pick things carefully.
            hub.linear_mixing(mix=1)
            self.assertAlmostEqual(np.abs(hub.nup - 1/3).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 7/4/3).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 4, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 7, msg=msg2)
            self.assertEqual(hub.nelectup, 4, msg=msg2)
            self.assertEqual(hub.nelectdown, 7, msg=msg2)
        with self.subTest(test="U=20"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=1, ncols=1, u=20)
            hub.set_electrons(nup=[1,1,0], ndown=[0,0,1])
            hub.set_kmesh(5,5)
            hub.linear_mixing(rdiff=1e-8)
            # Equality here will be measured quite loosely.
            n1 = 0.99543792
            n2 = 0.9889922
            self.assertAlmostEqual(np.abs(hub.nup - [n1, n1, (1-n1)*2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(np.abs(hub.ndown - [(1-n2)/2, (1-n2)/2, n2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(hub.nup.sum(), 2, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 1, msg=msg2)
            self.assertEqual(hub.nelectup, 2, msg=msg2)
            self.assertEqual(hub.nelectdown, 1, msg=msg2)
            self.assertLessEqual(hub.residual(), 1e-8, msg=msg3)
        with self.subTest(test="Full nup"):
            # With one electron channel full, it must be uniform.
            # The other one must be uniform too.
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=10)
            # The choice of ndown=1 and Gamma point only is deliberate, otherwise
            # degeneracies interfere with finding the solution.
            hub.set_electrons(nup=12, ndown=1, method='random')
            hub.set_kmesh(1,1)
            hub.linear_mixing(rdiff=1e-9)
            self.assertAlmostEqual(np.abs(hub.nup - 1).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 1/12).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 12, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 1, msg=msg2)
            self.assertEqual(hub.nelectup, 12, msg=msg2)
            self.assertEqual(hub.nelectdown, 1, msg=msg2)
            self.assertLessEqual(hub.residual(), 1e-9, msg=msg3)
    #
    def test_linear_mixing_T(self):
        # At finite T, failure to converge due to degeneracies is no longer a problem.
        # But, electrons can now move between the up and down channels.
        msg1 = "Density did not have expected value."
        msg2 = "Number of electrons changed."
        msg3 = "Residual not below expected threshold."
        with self.subTest(test="U=0"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=0, allow_fractions=True)
            hub.set_electrons(nup=4, ndown=3, method='random')
            hub.set_kmesh(5,5)
            hub.linear_mixing(mix=1, T=0.1)
            self.assertAlmostEqual(np.abs(hub.nup - 3.5/12).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 3.5/12).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 3.5, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 3.5, msg=msg2)
            self.assertAlmostEqual(hub.nelectup, 3.5, msg=msg2)
            self.assertAlmostEqual(hub.nelectdown, 3.5, msg=msg2)
        with self.subTest(test="Ferromagnetic"):
            # With a very high U and appropriate low electron density
            # and a nudge in the right direction, it should converge to a FM state
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=20, allow_fractions=True)
            hub.set_electrons(nup=4, ndown=2, method='uniform')
            hub.set_kmesh(5,5)
            hub.linear_mixing(T=0.1, rdiff=1e-8)
            self.assertAlmostEqual(np.abs(hub.nup - 0.5).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 0).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 6, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 0, msg=msg2)
            self.assertAlmostEqual(hub.nelectup, 6, msg=msg2)
            self.assertAlmostEqual(hub.nelectdown, 0, msg=msg2)
            self.assertLessEqual(hub.residual(T=0.1), 1e-8, msg=msg3)
        with self.subTest(test="U=20"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=1, ncols=1, u=20, allow_fractions=True)
            hub.set_electrons(nup=[1,0.9,0], ndown=[0,0.1,1])
            hub.set_kmesh(5,5)
            hub.linear_mixing(rdiff=1e-8, T=0.1)
            # Equality here will be measured quite loosely.
            n1 = 0.99543792
            n2 = 0.98899219
            self.assertAlmostEqual(np.abs(hub.nup - [n1, n1, (1-n1)*2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(np.abs(hub.ndown - [(1-n2)/2, (1-n2)/2, n2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(hub.nup.sum(), 2, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 1, msg=msg2)
            self.assertAlmostEqual(hub.nelectup, 2, msg=msg2)
            self.assertAlmostEqual(hub.nelectdown, 1, msg=msg2)
            self.assertLessEqual(hub.residual(0.1), 1e-8, msg=msg3)
    #
    def test_pulay_mixing(self):
        msg1 = "Density did not have expected value."
        msg2 = "Number of electrons changed."
        msg3 = "Residual not below expected threshold."
        with self.subTest(test="U=20"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=1, ncols=1, u=20)
            hub.set_electrons(nup=[1,1,0], ndown=[0,0,1])
            hub.set_kmesh(5,5)
            hub.pulay_mixing(rdiff=1e-8)
            # Equality here will be measured quite loosely.
            n1 = 0.99543792
            n2 = 0.9889922
            self.assertAlmostEqual(np.abs(hub.nup - [n1, n1, (1-n1)*2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(np.abs(hub.ndown - [(1-n2)/2, (1-n2)/2, n2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(hub.nup.sum(), 2, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 1, msg=msg2)
            self.assertEqual(hub.nelectup, 2, msg=msg2)
            self.assertEqual(hub.nelectdown, 1, msg=msg2)
            self.assertLessEqual(hub.residual(), 1e-8, msg=msg3)
        with self.subTest(test="Full nup"):
            # With one electron channel full, it must be uniform.
            # The other one must be uniform too.
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=10)
            # The choice of ndown=1 and Gamma point only is deliberate, otherwise
            # degeneracies interfere with finding the solution.
            hub.set_electrons(nup=12, ndown=1, method='random')
            hub.set_kmesh(1,1)
            hub.pulay_mixing(rdiff=1e-9)
            self.assertAlmostEqual(np.abs(hub.nup - 1).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 1/12).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 12, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 1, msg=msg2)
            self.assertEqual(hub.nelectup, 12, msg=msg2)
            self.assertEqual(hub.nelectdown, 1, msg=msg2)
            self.assertLessEqual(hub.residual(), 1e-9, msg=msg3)
    #
    def test_pulay_mixing_T(self):
        msg1 = "Density did not have expected value."
        msg2 = "Number of electrons changed."
        msg3 = "Residual not below expected threshold."
        with self.subTest(test="U=0"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=0, allow_fractions=True)
            hub.set_electrons(nup=4, ndown=3, method='random')
            hub.set_kmesh(5,5)
            hub.pulay_mixing(T=0.1, rdiff=1e-8)
            self.assertAlmostEqual(np.abs(hub.nup - 3.5/12).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 3.5/12).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 3.5, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 3.5, msg=msg2)
            self.assertAlmostEqual(hub.nelectup, 3.5, msg=msg2)
            self.assertAlmostEqual(hub.nelectdown, 3.5, msg=msg2)
        with self.subTest(test="Ferromagnetic"):
            # With a very high U and appropriate low electron density
            # and a nudge in the right direction, it should converge to a FM state
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=2, ncols=2, u=20, allow_fractions=True)
            hub.set_electrons(nup=5.8, ndown=0.2, method='uniform')
            hub.set_kmesh(5,5)
            hub.pulay_mixing(T=0.1, rdiff=1e-8)
            self.assertAlmostEqual(np.abs(hub.nup - 0.5).sum(), 0, msg=msg1)
            self.assertAlmostEqual(np.abs(hub.ndown - 0).sum(), 0, msg=msg1)
            self.assertAlmostEqual(hub.nup.sum(), 6, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 0, msg=msg2)
            self.assertAlmostEqual(hub.nelectup, 6, msg=msg2)
            self.assertAlmostEqual(hub.nelectdown, 0, msg=msg2)
            self.assertLessEqual(hub.residual(T=0.1), 1e-8, msg=msg3)
        with self.subTest(test="U=20"):
            hub = hubbard.kpoints.kagome.KagomeHubbardKPoints(nrows=1, ncols=1, u=20, allow_fractions=True)
            hub.set_electrons(nup=[1,0.9,0], ndown=[0,0.1,1])
            hub.set_kmesh(5,5)
            hub.pulay_mixing(rdiff=1e-8, T=0.1)
            # Equality here will be measured quite loosely.
            n1 = 0.99543792
            n2 = 0.98899219
            self.assertAlmostEqual(np.abs(hub.nup - [n1, n1, (1-n1)*2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(np.abs(hub.ndown - [(1-n2)/2, (1-n2)/2, n2]).sum(),
                                   0, msg=msg1, places=4)
            self.assertAlmostEqual(hub.nup.sum(), 2, msg=msg2)
            self.assertAlmostEqual(hub.ndown.sum(), 1, msg=msg2)
            self.assertAlmostEqual(hub.nelectup, 2, msg=msg2)
            self.assertAlmostEqual(hub.nelectdown, 1, msg=msg2)
            self.assertLessEqual(hub.residual(0.1), 1e-8, msg=msg3)
    
        
"""
Tests to write for hubbard.kpoints.kagome:
    set_electrons with other methods
all the things in base which I couldn't do before
    anderson_mixing
    band_structure
    density_of_states
    eigenstates
    fermi
    plot_bands
    plot_spin
    plot_charge
    plot_DOS
    plot_spincharge
"""

class TestBaseSubstrate(unittest.TestCase):
    """Tests hubbard.substrate.base"""
    # I can test adding, changing and removing substrates.
    # I can also test calling methods of those substrates.
    def test_init(self):
        hub = hubbard.substrate.base.HubbardSubstrate(nsites=3)
        self.assertEqual(hub.base_nsites, 3, msg="base_sites not initialised correctly.")
    #
    def test_add_substrate(self):
        """Tests if the method gets called correctly."""
        hub = hubbard.substrate.base.HubbardSubstrate(nsites=1)
        hub.add_substrate('square', 0, nx=1, ny=1)
        hub.add_substrate('triangle', 1, nx=1, ny=1)
        hub.add_substrate('dft', 2, nx=1, ny=2)
        self.assertEqual(len(hub.substrate_list), 3, msg="Have wrong number of substrates.")
        self.assertListEqual(hub.couplings, [0,1,2], msg="couplings not set correctly.")
        self.assertListEqual(hub.substrate_sites, [1,1,2], msg="substrate_sites not set correctly.")
        self.assertEqual(hub.nsites, 5, msg="nsites not set correctly.")
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup has the wrong length.")
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown has the wrong length.")
    #
    def test_remove_substrate(self):
        hub = hubbard.substrate.base.HubbardSubstrate(nsites=1)
        msg1 = "Have wrong number of substrates."
        msg2 = " was not set correctly."
        msg3 = " has the wrong length."
        hub.add_substrate('square', 0, nx=1, ny=1)
        hub.add_substrate('square', 1, nx=2, ny=1)
        hub.add_substrate('square', 2, nx=1, ny=3)
        hub.add_substrate('square', 3, nx=2, ny=2)
        self.assertEqual(len(hub.substrate_list), 4, msg=msg1)
        self.assertListEqual(hub.couplings, [0,1,2,3], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [1,2,3,4], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 11, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        # Remove the substrates one by one
        hub.remove_substrate(0)
        self.assertEqual(len(hub.substrate_list), 3, msg=msg1)
        self.assertListEqual(hub.couplings, [1,2,3], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [2,3,4], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 10, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.remove_substrate(-1)
        self.assertEqual(len(hub.substrate_list), 2, msg=msg1)
        self.assertListEqual(hub.couplings, [1,2], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [2,3], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 6, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.remove_substrate(1)
        self.assertEqual(len(hub.substrate_list), 1, msg=msg1)
        self.assertListEqual(hub.couplings, [1], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [2], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 3, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.remove_substrate(0)
        self.assertEqual(len(hub.substrate_list), 0, msg=msg1)
        self.assertListEqual(hub.couplings, [], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 1, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
    #
    def test_change_substrate(self):
        """Tests that change_substrate behaves as expected."""
        hub = hubbard.substrate.base.HubbardSubstrate(nsites=1)
        hub.add_substrate('square', 0, nx=1, ny=2)
        msg1 = "Have wrong number of substrates."
        msg2 = " was not set correctly."
        msg3 = " has the wrong length."
        self.assertEqual(len(hub.substrate_list), 1, msg=msg1)
        self.assertListEqual(hub.couplings, [0], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [2], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 3, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.change_substrate(0, coupling=1.5)
        self.assertEqual(len(hub.substrate_list), 1, msg=msg1)
        self.assertListEqual(hub.couplings, [1.5], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [2], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 3, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.change_substrate(0, nx=3, ny=1)
        self.assertEqual(len(hub.substrate_list), 1, msg=msg1)
        self.assertListEqual(hub.couplings, [1.5], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [3], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 4, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.change_substrate(0, subtype='triangle', nx=3, ny=1)
        self.assertEqual(len(hub.substrate_list), 1, msg=msg1)
        self.assertListEqual(hub.couplings, [1.5], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [3], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 4, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        hub.change_substrate(0, t=2, offset=-1)
        self.assertEqual(len(hub.substrate_list), 1, msg=msg1)
        self.assertListEqual(hub.couplings, [1.5], msg="couplings"+msg2)
        self.assertListEqual(hub.substrate_sites, [3], msg="substrate_sites"+msg2)
        self.assertEqual(hub.nsites, 4, msg="nsites"+msg2)
        self.assertEqual(len(hub.nup), hub.nsites, msg="nup"+msg3)
        self.assertEqual(len(hub.ndown), hub.nsites, msg="ndown"+msg3)
        self.assertEqual(hub.substrate_list[0].t, 2, msg="t"+msg2)
        self.assertEqual(hub.substrate_list[0].offset, -1, msg="offset"+msg2)

class TestKagomeSubstrate(unittest.TestCase):
    def test_init(self):
        hub = hubbard.substrate.kagome.KagomeSubstrate()
        msg = " not initialised correctly."
        self.assertEqual(hub.base_nsites, 3, msg="base_sites"+msg)
        self.assertEqual(hub.nsites, 3, msg="nsites"+msg)
        reclat = np.array([[sqrt(3)/2, 1/3], [-sqrt(3)/2, 1/2]])
        self.assertAlmostEqual(np.abs(hub.reclat - reclat).sum(), 0, msg="reclat"+msg)
        pos = np.array([[0,0], [0.5,0], [0,0.5]]) # Fractional coords
        self.assertAlmostEqual(np.abs(hub.positions - pos).sum(), 0, msg="positions"+msg)
    #
    def test_set_kinetic(self):
        pass







if __name__ == "__main__":
    unittest.main()
