#!/usr/bin/python3

import unittest

import numpy as np

import hubbard.kpoints.base

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

"""
Other things to test:
    _potential
    _chemical_potential_from_states
    _energy_from_states
    set_kmesh
    set_u
    set_mag
    toggle_allow_fractions
    set_electrons with allow_fractions=True
"""

if __name__ == "__main__":
    unittest.main()
