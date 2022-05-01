from freesasa import *
import unittest
import math
import os
import faulthandler

# this class tests using derived classes to create custom Classifiers
class DerivedClassifier(Classifier):
    purePython = True

    def classify(self,residueName,atomName):
        return 'bla'

    def radius(self,residueName,atomName):
        return 10

class FreeSASATestCase(unittest.TestCase):
    def testParameters(self):
        d = Parameters.defaultParameters
        p = Parameters()
        self.assertEqual(p.algorithm(), LeeRichards)
        self.assertEqual(p.algorithm(), d['algorithm'])
        self.assertEqual(p.probeRadius(), d['probe-radius'])
        self.assertEqual(p.nPoints(), d['n-points'])
        self.assertEqual(p.nSlices(), d['n-slices'])
        self.assertEqual(p.nThreads(), d['n-threads'])
        self.assertRaises(AssertionError,lambda: Parameters({'not-an-option' : 1}))
        self.assertRaises(AssertionError,lambda: Parameters({'n-slices' : 50, 'not-an-option' : 1}))
        self.assertRaises(AssertionError,lambda: Parameters({'not-an-option' : 50, 'also-not-an-option' : 1}))

        p.setAlgorithm(ShrakeRupley)
        self.assertEqual(p.algorithm(), ShrakeRupley)
        p.setAlgorithm(LeeRichards)
        self.assertEqual(p.algorithm(), LeeRichards)
        self.assertRaises(AssertionError,lambda: p.setAlgorithm(-10))

        p.setProbeRadius(1.5)
        self.assertEqual(p.probeRadius(), 1.5)
        self.assertRaises(AssertionError,lambda: p.setProbeRadius(-1))

        p.setNPoints(20)
        self.assertEqual(p.nPoints(), 20)
        self.assertRaises(AssertionError,lambda: p.setNPoints(0))

        p.setNSlices(10)
        self.assertEqual(p.nSlices(), 10)
        self.assertRaises(AssertionError,lambda: p.setNSlices(0))

        p.setNThreads(2)
        self.assertEqual(p.nThreads(), 2)
        self.assertRaises(AssertionError, lambda: p.setNThreads(0))

    def testResult(self):
        r = Result()
        self.assertRaises(AssertionError,lambda: r.totalArea())
        self.assertRaises(AssertionError,lambda: r.atomArea(0))

    def testClassifier(self):
        c = Classifier()
        self.assertTrue(c._isCClassifier())
        self.assertEqual(c.classify("ALA"," CB "), apolar)
        self.assertEqual(c.classify("ARG"," NH1"), polar)
        self.assertEqual(c.radius("ALA"," CB "), 1.88)

        setVerbosity(silent)
        self.assertRaises(Exception,lambda: Classifier("lib/tests/data/err.config"))
        self.assertRaises(IOError,lambda: Classifier(""))
        setVerbosity(normal)

        c = Classifier("lib/tests/data/test.config")
        self.assertEqual(c.classify("AA","aa"), "Polar")
        self.assertEqual(c.classify("BB","bb"), "Apolar")
        self.assertEqual(c.radius("AA","aa"), 1.0)
        self.assertEqual(c.radius("BB","bb"), 2.0)

        c = Classifier("lib/share/oons.config")
        self.assertEqual(c.radius("ALA"," CB "), 2.00)

        c = DerivedClassifier()
        self.assertFalse(c._isCClassifier())
        self.assertEqual(c.radius("ALA"," CB "), 10)
        self.assertEqual(c.radius("ABCDEFG","HIJKLMNO"), 10)
        self.assertEqual(c.classify("ABCDEFG","HIJKLMNO"), "bla")

    def testStructure(self):
        self.assertRaises(IOError,lambda: Structure("xyz#$%"))
        setVerbosity(silent)
        # test any file that's not a PDB file
        self.assertRaises(Exception,lambda: Structure("lib/tests/data/err.config"))
        self.assertRaises(Exception,lambda: Structure("lib/tests/data/empty.pdb"))
        self.assertRaises(Exception,lambda: Structure("lib/tests/data/empty_model.pdb"))
        setVerbosity(normal)

        s = Structure("lib/tests/data/1ubq.pdb")
        self.assertEqual(s.nAtoms(), 602)
        self.assertEqual(s.radius(1), 1.88)
        self.assertEqual(s.chainLabel(1), 'A')
        self.assertEqual(s.atomName(1), ' CA ')
        self.assertEqual(s.residueName(1), 'MET')
        self.assertEqual(s.residueNumber(1), '   1 ')

        s2 = Structure("lib/tests/data/1ubq.pdb",Classifier("lib/share/oons.config"))
        self.assertEqual(s.nAtoms(), 602)
        self.assertAlmostEqual(s2.radius(1), 2.0, delta=1e-5)

        s2 = Structure("lib/tests/data/1ubq.pdb",Classifier("lib/share/protor.config"))
        for i in range (0,601):
            self.assertAlmostEqual(s.radius(i), s2.radius(i), delta=1e-5)

        self.assertRaises(Exception,lambda: Structure("lib/tests/data/1ubq.pdb","lib/tests/data/err.config"))

        s = Structure()
        s.addAtom(' CA ','ALA','   1','A',1,1,1)
        self.assertEqual(s.nAtoms(), 1)
        self.assertEqual(s.atomName(0), ' CA ')
        self.assertEqual(s.residueName(0), 'ALA')
        self.assertEqual(s.residueNumber(0), '   1')
        self.assertEqual(s.chainLabel(0), 'A')
        self.assertEqual(s.nAtoms(), 1)
        x, y, z = s.coord(0)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)
        self.assertEqual(z, 1)
        s.addAtom(' CB ','ALA',2,'A',2,1,1)
        self.assertEqual(s.nAtoms(), 2)
        self.assertEqual(s.residueNumber(1), '2')

        # reinitialize s and test addAtoms function
        s = Structure()
        s.addAtoms([' CA ',' CB '], ['ALA','ALA'],['   1',2],['A','A'],[1,2],[1,1],[1,1])
        self.assertEqual(s.nAtoms(), 2)
        self.assertEqual(s.residueNumber(1), '2')

        self.assertRaises(AssertionError, lambda: s.atomName(3))
        self.assertRaises(AssertionError, lambda: s.residueName(3))
        self.assertRaises(AssertionError, lambda: s.residueNumber(3))
        self.assertRaises(AssertionError, lambda: s.chainLabel(3))
        self.assertRaises(AssertionError, lambda: s.coord(3))
        self.assertRaises(AssertionError, lambda: s.radius(3))

        s.setRadiiWithClassifier(Classifier())
        self.assertEqual(s.radius(0), 1.88)
        self.assertEqual(s.radius(1), 1.88)

        s.setRadiiWithClassifier(DerivedClassifier())
        self.assertEqual(s.radius(0), 10.0)
        self.assertEqual(s.radius(1), 10.0)

        s.setRadii([1.0,3.0])
        self.assertEqual(s.radius(0), 1.0)
        self.assertEqual(s.radius(1), 3.0)

        s.setRadius(0, 10.0)
        self.assertEqual(s.radius(0), 10.0);

        self.assertRaises(AssertionError,lambda: s.setRadius(2,10));
        self.assertRaises(AssertionError,lambda: s.setRadii([1]))
        self.assertRaises(AssertionError,lambda: s.setRadii([1,2,3]))

        self.assertRaises(AssertionError,lambda: s.atomName(2))
        self.assertRaises(AssertionError,lambda: s.residueName(2))
        self.assertRaises(AssertionError,lambda: s.residueNumber(2))
        self.assertRaises(AssertionError,lambda: s.chainLabel(2))

        setVerbosity(nowarnings)
        s = Structure("lib/tests/data/1d3z.pdb",None,{'hydrogen' : True})
        self.assertEqual(s.nAtoms(), 1231)

        s = Structure("lib/tests/data/1d3z.pdb",None,{'hydrogen' : True, 'join-models' : True})
        self.assertEqual(s.nAtoms(), 12310)

        s = Structure("lib/tests/data/1ubq.pdb",None,{'hetatm' : True})
        self.assertEqual(s.nAtoms(), 660)

        s = Structure("lib/tests/data/1d3z.pdb",None,{'hydrogen' : True, 'skip-unknown' : True})
        self.assertEqual(s.nAtoms(), 602)

        setVerbosity(silent)
        self.assertRaises(Exception, lambda : Structure("lib/tests/data/1d3z.pdb", None, {'hydrogen' : True, 'halt-at-unknown' : True}))
        setVerbosity(normal)

        s = Structure(options = { 'halt-at-unknown': True })
        setVerbosity(silent)
        self.assertRaises(Exception, lambda: s.addAtom(' XX ','ALA','   1','A',1,1,1))
        setVerbosity(normal)

        s = Structure(options = { 'skip-unknown': True })
        setVerbosity(silent)
        s.addAtom(' XX ','ALA','   1','A',1,1,1)
        self.assertEqual(s.nAtoms(), 0)
        setVerbosity(normal)

        s = Structure(classifier = Classifier.getStandardClassifier("naccess"))
        s.addAtom(' CA ', 'ALA','   1','A',1,1,1)
        self.assertEqual(s.radius(0), 1.87)

    def testStructureArray(self):
        # default separates chains, only uses first model (129 atoms per chain)
        ss = structureArray("lib/tests/data/2jo4.pdb")
        self.assertEqual(len(ss), 4)
        for s in ss:
            self.assertEqual(s.nAtoms(), 129)

        # include all models, separate chains, and include hydrogen and hetatm (286 atoms per chain)
        setVerbosity(nowarnings)
        ss = structureArray("lib/tests/data/2jo4.pdb",{'separate-models' : True,
                                                 'hydrogen' : True,
                                                 'hetatm' : True,
                                                 'separate-chains' : True})
        self.assertEqual(len(ss), 4*10)
        for s in ss:
            self.assertEqual(s.nAtoms(), 286)

        # include all models, and include hydrogen and hetatm (286 atoms per chain)
        ss = structureArray("lib/tests/data/2jo4.pdb",{'separate-models' : True,
                                             'hydrogen' : True,
                                             'hetatm' : True})
        self.assertEqual(len(ss), 10)
        for s in ss:
            self.assertEqual(s.nAtoms(), 286*4)
        setVerbosity(normal)

        # check that the structures initialized this way can be used for calculations
        ss = structureArray("lib/tests/data/1ubq.pdb")
        self.assertEqual(len(ss), 1)
        self.assertEqual(ss[0].nAtoms(), 602)
        result = calc(ss[0],Parameters({'algorithm' : ShrakeRupley}))
        self.assertAlmostEqual(result.totalArea(), 4834.716265, delta=1e-5)

        # Test exceptions
        setVerbosity(silent)
        self.assertRaises(AssertionError,lambda: structureArray(None))
        self.assertRaises(IOError,lambda: structureArray(""))
        self.assertRaises(Exception,lambda: structureArray("lib/tests/data/err.config"))
        self.assertRaises(AssertionError,lambda: structureArray("lib/tests/data/2jo4.pdb",{'not-an-option' : True}))
        self.assertRaises(AssertionError,
                          lambda: structureArray("lib/tests/data/2jo4.pdb",
                                                 {'not-an-option' : True, 'hydrogen' : True}))
        self.assertRaises(AssertionError,
                          lambda: structureArray("lib/tests/data/2jo4.pdb",
                                                 {'hydrogen' : True}))
        setVerbosity(normal)

    def testCalc(self):
        # test default settings
        structure = Structure("lib/tests/data/1ubq.pdb")
        result = calc(structure,Parameters({'algorithm' : ShrakeRupley}))
        self.assertAlmostEqual(result.totalArea(), 4834.716265, delta=1e-5)
        sasa_classes = classifyResults(result,structure)
        self.assertAlmostEqual(sasa_classes['Polar'], 2515.821238, delta=1e-5)
        self.assertAlmostEqual(sasa_classes['Apolar'], 2318.895027, delta=1e-5)

        # test residue areas
        residueAreas = result.residueAreas()
        a76 = residueAreas['A']['76']
        self.assertEqual(a76.residueType, "GLY")
        self.assertEqual(a76.residueNumber, "76")
        self.assertTrue(a76.hasRelativeAreas)
        self.assertAlmostEqual(a76.total, 142.1967898, delta=1e-5)
        self.assertAlmostEqual(a76.mainChain, 142.1967898, delta=1e-5)
        self.assertAlmostEqual(a76.sideChain, 0, delta=1e-5)
        self.assertAlmostEqual(a76.polar, 97.297889, delta=1e-5)
        self.assertAlmostEqual(a76.apolar, 44.898900, delta=1e-5)
        self.assertAlmostEqual(a76.relativeTotal, 1.75357, delta=1e-4)
        self.assertAlmostEqual(a76.relativeMainChain, 1.75357, delta=1e-4)
        self.assertTrue(math.isnan(a76.relativeSideChain))
        self.assertAlmostEqual(a76.relativePolar, 2.17912, delta=1e-4)
        self.assertAlmostEqual(a76.relativeApolar, 1.23213, delta=1e-4)

        # test L&R
        result = calc(structure,Parameters({'algorithm' : LeeRichards, 'n-slices' : 20}))
        sasa_classes = classifyResults(result,structure)
        self.assertAlmostEqual(result.totalArea(), 4804.055641, delta=1e-5)
        self.assertAlmostEqual(sasa_classes['Polar'], 2504.217302, delta=1e-5)
        self.assertAlmostEqual(sasa_classes['Apolar'], 2299.838339, delta=1e-5)

        # test extending Classifier with derived class
        sasa_classes = classifyResults(result,structure,DerivedClassifier())
        self.assertAlmostEqual(sasa_classes['bla'], 4804.055641, delta=1e-5)

        ## test calculating with user-defined classifier ##
        classifier = Classifier("lib/share/oons.config")
        # classifier passed to assign user-defined radii, could also have used setRadiiWithClassifier()
        structure = Structure("lib/tests/data/1ubq.pdb",classifier)
        result = calc(structure,Parameters({'algorithm' : ShrakeRupley}))
        self.assertAlmostEqual(result.totalArea(), 4779.5109924, delta=1e-5)
        sasa_classes = classifyResults(result,structure,classifier) # classifier passed to get user-classes
        self.assertAlmostEqual(sasa_classes['Polar'], 2236.9298941, delta=1e-5)
        self.assertAlmostEqual(sasa_classes['Apolar'], 2542.5810983, delta=1e-5)


    def testCalcCoord(self):
        # one unit sphere
        radii = [1]
        coord = [0,0,0]
        parameters = Parameters()
        parameters.setNSlices(5000)
        parameters.setProbeRadius(0)
        parameters.setNThreads(1)
        result = calcCoord(coord, radii, parameters)
        self.assertAlmostEqual(result.totalArea(), 4*math.pi, delta=1e-3)

        # two separate unit spheres
        radii = [1,1]
        coord = [0,0,0, 4,4,4]
        result = calcCoord(coord, radii, parameters)
        self.assertAlmostEqual(result.totalArea(), 2*4*math.pi, delta=1e-3)

        self.assertRaises(AssertionError,
                          lambda: calcCoord(radii, radii))

    def testSelectArea(self):
        structure = Structure("lib/tests/data/1ubq.pdb")
        result = calc(structure,Parameters({'algorithm' : ShrakeRupley}))
        # will only test that this gets through to the C interface,
        # extensive checking of the parser is done in the C unit tests
        selections = selectArea(('s1, resn ala','s2, resi 1'),structure,result)
        self.assertAlmostEqual(selections['s1'], 118.35, delta=0.1)
        self.assertAlmostEqual(selections['s2'], 50.77, delta=0.1)

    def testBioPDB(self):
        try:
            from Bio.PDB import PDBParser
        except ImportError:
            print("Can't import Bio.PDB, tests skipped")
            pass
        else:
            parser = PDBParser(QUIET=True)
            bp_structure = parser.get_structure("29G11","lib/tests/data/1a0q.pdb")
            s1 = structureFromBioPDB(bp_structure)
            s2 = Structure("lib/tests/data/1a0q.pdb")
            self.assertEqual(s1.nAtoms(), s2.nAtoms())

            # make sure we got the insertion code
            self.assertEqual(s1.residueNumber(2286), '82A')

            for i in range(0, s2.nAtoms()):
                self.assertEqual(s1.radius(i), s2.radius(i))

                # there can be tiny errors here
                self.assertAlmostEqual(s1.coord(i)[0], s2.coord(i[0]), delta=1e-5)
                self.assertAlmostEqual(s1.coord(i)[1], s2.coord(i[1]), delta=1e-5)
                self.assertAlmostEqual(s1.coord(i)[2], s2.coord(i[2]), delta=1e-5)

                # whitespace won't match
                self.assertIn(s1.residueNumber(i), s2.residueNumber(i))

            # because Bio.PDB structures will have slightly different
            # coordinates (due to rounding errors) we set the
            # tolerance as high as 1e-3
            result = calc(s1, Parameters({'algorithm' : LeeRichards, 'n-slices' : 20}))
            self.assertAlmostEqual(result.totalArea(), 18923.280586, delta=1e-3)
            sasa_classes = classifyResults(result, s1)
            self.assertAlmostEqual(sasa_classes['Polar'], 9143.066411, delta=1e-3)
            self.assertAlmostEqual(sasa_classes['Apolar'], 9780.2141746, delta=1e-3)
            residue_areas = result.residueAreas()
            self.assertAlmostEqual(residue_areas['L']['2'].total, 43.714, delta=1e-2)

            faulthandler.enable()
            result, sasa_classes = calcBioPDB(bp_structure, Parameters({'algorithm' : LeeRichards, 'n-slices' : 20}))
            self.assertAlmostEqual(result.totalArea(), 18923.280586, delta=1e-3)
            self.assertAlmostEqual(sasa_classes['Polar'], 9143.066411, delta=1e-3)
            self.assertAlmostEqual(sasa_classes['Apolar'], 9780.2141746, delta=1e-3)
            residue_areas = result.residueAreas()
            self.assertAlmostEqual(residue_areas['L']['2'].total, 43.714, delta=1e-2)

            options = {'hetatm': False,
                       'hydrogen': False,
                       'join-models': False,
                       'skip-unknown': True,
                       'halt-at-unknown': False}
            classifier = Classifier()

            bp_structure = parser.get_structure("PDB_WITH_HYDROGENS","lib/tests/data/1d3z.pdb")
            fs_structure = Structure("lib/tests/data/1d3z.pdb", classifier, options)

            fsfrombp = structureFromBioPDB(bp_structure, classifier, options)
            self.assertEqual(fs_structure.nAtoms(), fsfrombp.nAtoms())

if __name__ == '__main__':
    # make sure we're in the right directory (if script is called from
    # outside the directory)
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)
    unittest.main()
