### Plot for threshold variation
['from', 'director', 'title']; 0.00, (1,6) ----------- 70.19
['from', 'director', 'title']; 0.05, (1,6) ----------- 70.06
['from', 'director', 'title']; 0.10, (1,6) ----------- 69.94
['from', 'director', 'title']; 0.15, (1,6) ----------- 69.44
['from', 'director', 'title']; 0.20, (1,6) ----------- 69.57
['from', 'director', 'title']; 0.25, (1,6) ----------- 68.94
['from', 'director', 'title']; 1.00, (1,6) ----------- 66.71

### Plot for ngram variation
['from', 'director', 'title']; 0.00, (1,3) ----------- 70.19
['from', 'director', 'title']; 0.00, (1,4) ----------- 69.94
['from', 'director', 'title']; 0.00, (1,5) ----------- 70.31 F1-score --- 0.70
['from', 'director', 'title']; 0.00, (1,6) ----------- 70.19
['from', 'director', 'title']; 0.00, (1,7) ----------- 70.19
['from', 'director', 'title']; 0.00, (1,8) ----------- 70.06
['from', 'director', 'title']; 0.00, (1,9) ----------- 70.19

### Without lemma
['from', 'director', 'title']; 0.00, (1,5) ----------- 69.19 F1-score --- 0.69

### Without fields to combine
[]; 0.00, (1,5) -------------------------------------- 67.70 F1-score --- 0.68

### Without the genre names as stop words
['from', 'director', 'title']; 0.00, (1,5) ----------- 69.81 F1-score --- 0.70

### Without stop words at all
['from', 'director', 'title']; 1.00, (1,5) ----------- 66.83 F1-score --- 0.67

### Without data augmentation
['from', 'director', 'title']; 0.00, (1,5) ----------- 67.58 F1-score --- 0.67