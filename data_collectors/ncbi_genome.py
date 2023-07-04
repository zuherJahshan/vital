import os
import sys
import time
import json
from typing import List

__ORIG_WD__ = os.getcwd()
os.chdir("../utils/")

from utils import *

os.chdir(__ORIG_WD__)



Taxon = str

def taxaToJsonFileName(string: str) -> str:
    """
    Converts a string to a valid file name.
    """
    return '-'.join(string.lower().split(' ')) + '.json'

def printProgresBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    Args:
        iteration: current iteration (Int)
        total: total iterations (Int)
        prefix: prefix string (Str)
        suffix: suffix string (Str)
        decimals: positive number of decimals in percent complete (Int)
        length: character length of bar (Int)
        fill: bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

def getTaxonListFromFile(fileName: str) -> List[Taxon]:
    """
    Reads a file and returns a list of taxon names.
    """
    taxonList = []
    with open(fileName, 'r') as f:
        for line in f:
            taxonList.append(line.strip())
    return taxonList

class NCBIGenome(object):

    class Status:
        success = "success"
        failed = "failed"
    
    class InfoMessages:
        success = "Accessions found"
        taxonNotFound = "Taxon not found"

        @staticmethod
        def failedCustomMessage(message: str):
            return f"Failed: {message}"

        
    def __init__(self, outputdir: str = "data/ncbi_genome"):
        # check if paths are absolute, if not add __ORIG_WD__ to them
        if not os.path.isabs(outputdir):
            self.metadataDir = f"{__ORIG_WD__}/{outputdir}/metadata"
            self.dataDir = f"{__ORIG_WD__}/{outputdir}/accessions"
        else:
            self.metadataDir = f"{outputdir}/metadata"
            self.dataDir = f"{outputdir}/accessions"


    def downloadAccessions(
        self,
        taxonNameList: List[Taxon],
        numberOfAccessionsPerTaxon: int = 1
    ) -> None:
        """
        This function will download the accessions of the given taxon names.
        It will return a nested dictionary that is structored as follows:
        {
            taxonName: {
                status: "success" | "failed",
                accessions: [accession1, accession2, ...]
                infoMessage: "Taxon not found" | "Accessions not found" | "Accessions found" ...
            }
        }
        """

        # iterate over the taxon names, add them to the retDict and download their metadata
        retDict = {}
        accessions = []
        metadataDownloadRetStatus = self._downloadMetadata(taxonNameList) # This call will never fail.
        for taxonName in taxonNameList:
            if metadataDownloadRetStatus[taxonName] != NCBIGenome.InfoMessages.success:
                retDict.update({taxonName: {
                    "status": NCBIGenome.Status.failed,
                    "accessions": [],
                    "infoMessage": metadataDownloadRetStatus[taxonName]
                }})
            else:
                taxonAccessions = self._getAccessionsFromMetadataFile(self._getMetadataFileName(taxonName))
                retDict.update({taxonName: {
                    "status": NCBIGenome.Status.success,
                    "accessions": taxonAccessions,
                    "infoMessage": NCBIGenome.InfoMessages.success
                }})
                for accession in taxonAccessions[:numberOfAccessionsPerTaxon]:
                    if os.path.exists(f"{self.dataDir}/{accession}"):
                        continue
                    accessions.append(accession)
        if len(accessions) > 0:
            self._downloadAccessions(accessions)
            

    def getLocalAccessions(self, taxon: Taxon) -> List[str]:
        # get all accessions of the taxon name. Then check which is available locally and return it.
        accessions = self._getAccessionsFromMetadataFile(self._getMetadataFileName(taxon))
        filepaths = []
        for accession in accessions:
            if os.path.exists(f"{self.dataDir}/{accession}"):
                for dirpath,_ ,filenames in os.walk(f"{self.dataDir}/{accession}"):
                    for filename in filenames:
                        if is_fasta_file(filename):
                            filepaths.append(os.path.abspath(os.path.join(dirpath, filename)))
        return filepaths
    
    def getLocalTaxa(self) -> List[str]:
        # get all taxons according to Metadata files
        return [self._getTaxonFromFileName(filename) for filename in os.listdir(f"{self.metadataDir}")]

    
    ##############################
    ### Private helper methods ###
    ##############################
    def _getMetadataFileName(self, taxonName: str) -> str:
        return f"{self.metadataDir}/{taxaToJsonFileName(taxonName)}"
    
    def _getTaxonFromFileName(self, fileName: str) -> str:
        return fileName.split('/')[-1].split('.')[0]

    def _downloadMetadata(
        self,
        taxonList: List[Taxon]
    ) -> None:
        # create the output dir iun case it does not exist
        if not os.path.exists(self.metadataDir):
            os.makedirs(self.metadataDir)

        # For each taxon name, download the metadata that states the different assemblies.
        retDict = {}
        for taxonName in taxonList:
            outputFile = self._getMetadataFileName(taxonName)
            # Check if the metadata file already exists, if not download it.
            if not os.path.exists(outputFile):
                command = f"datasets summary genome taxon '{taxonName}' --assembly-level complete\
                    > {outputFile}"
                os.system(command)
            # Check validity of downloaded file
            try:
                accessions = self._getAccessionsFromMetadataFile(outputFile)
            except Exception as e:
                retDict.update({taxonName: NCBIGenome.InfoMessages.failedCustomMessage(str(e))})
                continue
            if len(accessions) == 0:
                retDict.update({taxonName: NCBIGenome.InfoMessages.taxonNotFound})
            else:
                retDict.update({taxonName: NCBIGenome.InfoMessages.success})
        return retDict


    def _getAccessionsFromMetadataFile(self, filename):
        """
        This function will return a list of accessions from a metadata file.
        """
        if not os.path.exists(filename):
            return []
        with open(filename, 'r') as f:
            # write json file to dict. If returns an error, thrown an exception
            try:
                metadata = json.load(f)
            except:
                raise Exception("Could not load metadata file.")
            # get the accessions from the metadata
            if "reports" in metadata and "accession" in metadata["reports"][0]:
                return [report["accession"] for report in metadata["reports"]]
            elif "total_count" in metadata and metadata["total_count"] == 0:
                return []
            else:
                # throw an exception if the metadata file does not contain the accessions
                raise Exception(f"Metadata file \"{filename}\" does not contain accessions, ")
            
    # read the json files and translate them into objects, the structure of the json files is as follows:
    # {reports: [List of accessions and their metadata], totalcount: [Total number of accessions]}
    # Each accession has the following structure:
    # { accession: <id> , assembly_level: <level>, bioproject: <id>, biosample: <id>, wgs_master: <id>,
    #   ...}
    # The fields that are important to us are the accession.

    def _downloadAccessions(self, accessions_to_download):
        # create the output directory if it does not exist
        output_dirname = self.dataDir
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
        
        # Check if there is accessions to download
        if len(accessions_to_download) == 0:
            return
        
        # Download the accessions
        
        # Create tmp folder and tmp zip file for download and unzip
        tmp_zip_filename = f"tmp.zip"
        download_folder = "ncbi_dataset/data/"
        
        # Download the accessions in chunks of 100, to make the request successfull
        # TBD - check if the request is successfull, if not try again with a smaller chunk size
        download_chunk_size = 100

        # Download the chunks of accessions, unzip them and move them to the output directory
        for i in range(0, len(accessions_to_download) + download_chunk_size, download_chunk_size):
            chunk_to_download = accessions_to_download[i: i+download_chunk_size]
            if chunk_to_download == []:
                break
            os.system(f"nohup datasets download genome accession \
                    {' '.join(accessions_to_download[i: i+download_chunk_size])} --filename {tmp_zip_filename}")
            os.system(f"nohup unzip -o {tmp_zip_filename} && mv {download_folder}/* {output_dirname}/")
            os.system(f"rm -rf {download_folder} && rm {tmp_zip_filename}")
            print(f"Done downloading {i} accessions.")
        return