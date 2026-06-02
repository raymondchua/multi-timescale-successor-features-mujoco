import csv
import os
import argparse
import re


"""
Create a function call wandb_sync_in_folder to sync all wandb files in a folder
Loop through all the folders in the directory
For each folder, find the file that ends with _log.out
Open the file and go to the second last line of the file
Check if the line starts with "wand: wandb sync"
If it does, save it to a new variable called "wandb_sync" and print it
Remove the first seven characters from the line
Save the remaining line to a new variable called "wandb_sync"
Execute the line in the terminal
Else if does not start with "wand: wandb sync", print "No wandb sync command found" 
and save the line to a new variable called "no_wandb_sync"
"""


def wandb_sync_in_folder(folder_dir: str, direct_dir: bool) -> None:
    """
    Read a folder directory path and sync all wandb files in that folder.
    Arguments:
        folder_dir: a string, the path to a folder
    Returns:
        A list of lists, where each sublist is a row in the csv file.
    """
    successful_sync = []
    failed_sync = []

    print("folder_dir: ", folder_dir)

    # loop through all the folders in the directory
    for folder in os.listdir(folder_dir):
        # find the file that ends with _log.out

        # open the folder and check if there is a folder called "submitit"
        # if the folder does not exist, skip the folder
        # else open the folder and get the directory of the folder called "submitit"

        print("folder: ", folder)

        if direct_dir:
            submitit_dir = os.path.join(folder_dir, ".submitit")
        else:
            submitit_dir = os.path.join(folder_dir, folder, ".submitit")

        print("submitit_dir: ", submitit_dir)

        if not os.path.isdir(submitit_dir):
            continue
        else:
            for items in os.listdir(submitit_dir):

                # check if items is a folder
                if os.path.isdir(os.path.join(submitit_dir, items)):

                    # access the folder
                    for file in os.listdir(os.path.join(submitit_dir, items)):
                        # print file name
                        print("file: ", file)

                        if file.endswith("_log.out"):
                            # open the file
                            with open(
                                os.path.join(submitit_dir, items, file), "r"
                            ) as f:

                                # go to the last 10 lines of the file
                                for line in f.readlines()[-10:]:

                                    if line.startswith("wandb: wandb sync"):
                                        # if it does, save it to a new variable called "wandb_sync" and print it
                                        wandb_sync = line

                                        # if the string contains "/home/chuaraym/CRLMSF/CRLMSF/", replace it with /home/chuaraym/scratch/
                                        if (
                                            "/home/chuaraym/CRLMSF/CRLMSF/"
                                            in wandb_sync
                                        ):
                                            wandb_sync = wandb_sync.replace(
                                                "/home/chuaraym/CRLMSF/CRLMSF/",
                                                "/home/chuaraym/scratch/",
                                            )

                                        # print(wandb_sync)
                                        # remove the first seven characters from the line
                                        wandb_sync = wandb_sync[7:]
                                        # save the remaining line to a new variable called "wandb_sync"
                                        wandb_sync = wandb_sync.strip()

                                        # print the line
                                        print(
                                            "currently syncing: ", submitit_dir, "..."
                                        )
                                        print("")

                                        # execute the line in the terminal
                                        os.system(wandb_sync)
                                        # print row
                                        print("row: ", submitit_dir)
                                        successful_sync.append(submitit_dir)

                                        # print an empty line
                                        print("")

                                    # to handle irregular expressions
                                    elif "1mwandb sync " in line:

                                        # clean the line
                                        wandb_sync = re.sub(r'\x1b\[[0-9;]*m', '', line)

                                        # split the line using ":" as separator and keep the second substring
                                        wandb_sync = wandb_sync.split(":")[1]

                                        # execute the line in the terminal
                                        os.system(wandb_sync)
                                        # print row
                                        print("row: ", submitit_dir)
                                        successful_sync.append(submitit_dir)

                                        # print an empty line
                                        print("")

                                    # else if does not start with "wand: wandb sync", print "No wandb sync command found" and save the line to a new variable called "no_wandb_sync"
                                    else:
                                        # print job_id has failed to sync
                                        print("failed to sync: ", submitit_dir)
                                        failed_sync.append(submitit_dir)

                            # close the file
                            f.close()

        # break the loop if direct_dir is True to avoid looping through all the folders which are just empty folders
        # with seed numbers as folder name
        if direct_dir:
            break

    # check if the file successful_syncs.txt exists
    if os.path.exists("successful_syncs.txt"):
        # if it does, open and add to it
        with open("successful_syncs.txt", "a") as f:
            for sync in successful_sync:
                # add " - done" to the end of the sync
                sync = sync + " - done"
                f.write(sync + "\n")

        # close the file
        f.close()

    else:
        # if it does not, create it and add to it
        with open("successful_syncs.txt", "w") as f:
            for sync in successful_sync:
                # add " - done" to the end of the sync
                sync = sync + " - done"
                f.write(sync + "\n")

        # close the file
        f.close()

    # check if the file failed_syncs.txt exists
    if os.path.exists("failed_syncs.txt"):
        # if it does, open and add to it
        with open("failed_syncs.txt", "a") as f:
            for sync in failed_sync:
                # add " - failed" to the end of the sync
                sync = sync + " - failed"
                f.write(sync + "\n")

        # close the file
        f.close()

    else:
        # if it does not, create it and add to it
        with open("failed_syncs.txt", "w") as f:
            for sync in failed_sync:
                # add " - failed" to the end of the sync
                sync = sync + " - failed"
                f.write(sync + "\n")

        # close the file
        f.close()


def main():
    # create an argument parser object that takes a string name csv file path as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_dir", help="The path to the folder directory", required=True
    )
    parser.add_argument(
        "--direct_dir",
        help="Use this flag when the path to the direct directory is given. Default is False.",
        action="store_true",
    )
    args = parser.parse_args()
    wandb_sync_in_folder(folder_dir=args.folder_dir, direct_dir=args.direct_dir)


if __name__ == "__main__":
    main()
