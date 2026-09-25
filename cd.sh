
ssh subhanu@10.208.67.120

psswd: subhanu_server@123

cd /home/subhanu/GOT_Compression/IITD_GOT/Prasoon/

git clone https://github.com/TREX4096/Graph_of_Thought

cd /home/subhanu/GOT_Compression/IITD_GOT/Prasoon/Graph_of_Thought


cd ~/GOT_Compression/IITD_GOT/Prasoon/Graph_of_Thought
conda activate GOTComp-cu121
pip install -e ".[hpc]"
python scripts/generate_data.py --out data --seed 42 --n-samples 100 --small
