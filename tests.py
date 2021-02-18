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


if __name__ == "__main__":
    unittest.main()
