"""The login-node preflight must flag exactly the pinned files whose bytes changed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_contract_code as check


class Preflight(unittest.TestCase):
    def test_changed_and_missing_files_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'a.py').write_text('x = 1\n')
            (root/'b.py').write_text('y = 2\n')
            contract = root/'contract.json'
            contract.write_text(json.dumps(dict(code={n: check.sha256(root/n) for n in ('a.py', 'b.py')})))
            self.assertEqual(check.changed(contract, root), [])
            (root/'b.py').write_text('y = 3\n')
            (root/'a.py').unlink()
            self.assertEqual(check.changed(contract, root), ['a.py', 'b.py'])

    def test_matches_the_trainers_hash(self):
        from ms2_eval.io import sha256_file
        path = ROOT/'tools/iris_decoder_only.py'
        self.assertEqual(check.sha256(path), sha256_file(path))


if __name__ == '__main__':
    unittest.main()
