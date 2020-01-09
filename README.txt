For efficient pruning, the function do_pruning does the following:
1) Compares the length of the corresponding word from the dictionary. If the length is less than or equal to 4, then
it only compares the start to start and end to end distances of the template point with a certain threshold value.
2) If the length is greater than 4, it also considers some intermediate points among the sampled ones for comparison
so as to exclude unnecessary words after pruning.

After verification, it's observed that the actually typed word does occur in the words list obtained after pruning which
we can see in the console. Also, the total number of words in the pruned list is pretty less as compared to original
list which helps in speeding up the code. Later on, we calculate the shape and location scores of all valid words
obtained after pruning.

This implementation describes the standard trade-off between speed and accuracy. If we want more accuracy, then speed of
recognition decreases drastically. So we try to balance both here.

All the rest of implementation is the same as mentioned in the Shark2 research paper.

After pruning, we get the integration scores after adding the shape and location scores. We then output the first 6
words having the least integration score.