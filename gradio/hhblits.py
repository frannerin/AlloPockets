import tempfile, subprocess, re, sys
import pandas as pd
import numpy as np


mmseqs = "/home/fnerin/miniconda3/envs/mmseqs/bin/mmseqs"
db = "/data/fnerin/uniref30_2302/uniref30_2302_db"


class HHBlitsF:
    def __init__(self, cif):
        self._cif = cif

    features = [
        "hhblits"
    ]

    def _hhblits(self, seq):        
        with tempfile.TemporaryDirectory() as tmpdir:
            outf = f"{tmpdir}/hhm.hhm"
            
            fastaf = f"{tmpdir}/fasta.fasta"
            with open(fastaf, "w") as fasta:
                fasta.write(f">fasta\n{seq}\n") 

            subprocess.run(
                (
                    f"{mmseqs} createdb {fastaf} {tmpdir}/qdb -v 1 && "
                    + f"{mmseqs} search {tmpdir}/qdb {db} {tmpdir}/result {tmpdir}/tmp  --gpu 1 --num-iterations 3 -e 0.1 --max-seqs 20000 --prefilter-mode 1 -a -v 1 && " # -s 9 ignored with GPU
                    + f"{mmseqs} mvdb {tmpdir}/tmp/latest/profile_1 {tmpdir}/prof_res && "
                    + f"{mmseqs} lndb {tmpdir}/qdb_h {tmpdir}/prof_res_h && "
                    + f"{mmseqs} align {tmpdir}/prof_res {db} {tmpdir}/result {tmpdir}/result_realign -e 10 --max-accept 1000000 --alt-ali 10 -a -v 1 && "
                    + f"{mmseqs} filterresult {tmpdir}/qdb {db} {tmpdir}/result_realign {tmpdir}/result_filter --qid 0 --qsc 0.8 --diff 0 --max-seq-id 1.0 --filter-min-enable 100 -v 1 && "
                    + f"{mmseqs} result2msa {tmpdir}/qdb {db} {tmpdir}/result_filter {tmpdir}/result_msa --msa-format-mode 6 --filter-msa 1 --filter-min-enable 1000 --diff 3000 --qid 0.0,0.2,0.4,0.6,0.8,1.0 --qsc 0 --max-seq-id 0.95 -v 1 && "
                    + f"hhmake -i {tmpdir}/result_msa -o {outf} -v 1"
                ),
                shell=True, check=True
            )

            # HHM processing from moleculekit https://github.com/Acellera/moleculekit/blob/433b6d188edc9a405fccfe4550e65431972a13a9/moleculekit/tools/hhblitsprofile.py
            data = []
            seq = []
            with open(outf, "r") as fp:
                regex = re.compile("^\w\s\d+")
                starting = 0
                lines = fp.readlines()
                for i in range(len(lines)):
                    if lines[i].startswith("NULL"):
                        pieces = lines[i].split()
                        seq.append([pieces[0]])
                        data.append(
                            [2 ** (-int(x) / 1000) if x != "*" else 0 for x in pieces[1:21]]
                            + [0] * 10
                        )
                    if lines[i].startswith("HMM    A	C	D"):
                        col_desc = lines[i].split()[1:] + lines[i + 1].split()
                        starting = 1
                    if starting > 0:
                        starting += 1
                    if starting >= 4 and regex.match(lines[i]):
                        pieces = lines[i].split()
                        seq.append([pieces[0]])
                        d = [2 ** (-int(x) / 1000) if x != "*" else 0 for x in pieces[2:22]]
                        pieces = lines[i + 1].split()
                        d += [2 ** (-int(x) / 1000) if x != "*" else 0 for x in pieces[:7]]
                        d += [0.001 * int(x) for x in pieces[7:10]]
                        data.append(d)
        
            return pd.DataFrame(
                np.hstack((np.vstack(seq), np.vstack(data))), columns=["seq"] + col_desc
            )

    def hhblits(self):
        dfs = []
        res = self._cif.residues
        ents = pd.DataFrame(self._cif.cif.data["_entity_poly"], dtype=str)
        for entity_id, entity_res in res.groupby("label_entity_id"):
            seq = ents.query(f"entity_id == '{entity_id}'")["pdbx_seq_one_letter_code_can"].item().replace("\n", "")
            df = self._hhblits(seq)
            for asym_id in entity_res.label_asym_id.unique():
                dfs.append(
                    pd.concat(
                        [
                            pd.DataFrame({
                                "label_asym_id": [asym_id]*len(seq),
                                "label_seq_id": range(1, len(seq)+1)
                            }, dtype=str),
                            df.iloc[1:].drop(columns="seq").reset_index(drop=True)
                        ], 
                        axis=1
                    )
                )
        hhblits = pd.concat(dfs)
        return res.merge(
            hhblits,
            on=["label_asym_id", "label_seq_id"]
        )[hhblits.columns]





import sys, os

_, pdb, path = sys.argv

sys.path.append(__file__.rsplit("/", 2)[0] + "/training_data")
from utils.features_utils import calculate_features

if __name__ == '__main__':
    fc = HHBlitsF
    pdbpath = f"{path}/features/{pdb}"
    os.makedirs(pdbpath, exist_ok=True)
    file = f"{pdbpath}/{fc.__name__}.pkl"
    if not os.path.isfile(file):
        calculated = calculate_features(pdb, fc, file, path, path)
        assert calculated, f"Feature calculation failed: {fc.__name__}"