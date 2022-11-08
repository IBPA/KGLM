## Steps for extracting raw data
**Step 1.** Raw data file was downloaded from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52312). In order to save space in the repository, we removed files '*text_\*.txt*' and zipped it again at <code>[FB15K-237-modified.zip](./FB15K-237-modified.zip)</code>.

**Step 2.** (Optional) You may have to install unzip to run the following bash script.
```
sudo apt-get install unzip
```

**Step 3.** Extract data from the compressed file.
```
./extract_fb15k237.sh
```
