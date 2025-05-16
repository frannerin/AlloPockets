import os, pickle

with open("../../../training_data/8.Apos/Extra_set/features.pkl", "rb") as f:
    extras_featuresd = pickle.load(f)

len(extras_featuresd), extras_featuresd




import subprocess, pymol2

for pdb in extras_featuresd:
    pdb = pdb.upper()
    os.makedirs(pdb, exist_ok=True)

    # os.system(f"ln -s {os.getcwd().rsplit('/', 1)[0]}/structures/{pdb.lower()}.pdb {pdb}/{pdb}.pdb") # pdb file needs a HEADER first line for mkdssp to work
    pdbf = f"{pdb}/{pdb}.pdb"
    if not os.path.isfile(pdbf):
        with (
            open(f"{os.getcwd().rsplit('/', 1)[0]}/structures/{pdb.lower() if 'AF' not in pdb else pdb}.pdb", "r") as orig_pdbf,
            open(pdbf, "w") as f
        ):
            f.write(f"HEADER {pdb}\n")
            f.write(orig_pdbf.read())
    
    # os.system(f"./mkdssp-4.4.0-linux-x64 --output-format dssp {pdb}/{pdb}.pdb") # need to capture output
    dsspf = pdbf.replace(".pdb", ".dssp")
    if not os.path.isfile(dsspf):
        with open(dsspf, "w") as f:
            f.write(
                subprocess.run(["./mkdssp-4.4.0-linux-x64", "--output-format=dssp", pdbf], capture_output=True)
                .stdout.decode()
            )

    asnf = pdbf.replace(".pdb", ".asn")
    if not os.path.isfile(asnf):
        with pymol2.PyMOL() as pymol, open(pdbf.replace(".pdb", ".fasta"), "w") as f:#tempfile.NamedTemporaryFile("w+", suffix=".fasta") as f:
            pymol.cmd.load(pdbf, "prot")
            f.write(pymol.cmd.get_fastastr('prot'))
        
        os.system(f"cd nr && psiblast -query {os.getcwd()}/{pdbf.replace('.pdb', '.fasta')} -db nr -out {os.getcwd()}/{pdbf.replace('.pdb', '.out')} -num_iterations 3 -evalue 0.001 -outfmt 11 -out_pssm {os.getcwd()}/{asnf} -save_pssm_after_last_round -num_threads 10")