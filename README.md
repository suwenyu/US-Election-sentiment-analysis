# Twitter Sentiment Analysis of the 2016 U.S. Presidential Election

This project proposed to use different models with several pre-processing methods for sentiment prediction. Experiments conducted on two datasets which are Obama and Romney attest the effectiveness of different models.

### Result
/***
\begin{table}
 \caption{Effect of deep learning}
  \centering
  
  \begin{tabular}{lllllllll}
    \toprule
     & \multicolumn{4}{c}{Obama}      & \multicolumn{4}{c}{Romney}                 \\
    \cmidrule(r){2-5}              
    \cmidrule(r){6-9} 
    &    Accuracy &  \multicolumn{3}{c}{F1} & Accuracy & \multicolumn{3}{c}{F1} \\
    \midrule
    Naive Bayes & 57.54\% & 50.98\% & 57.60\% & \textbf{62.81\%} & 53.74\%  &  14.92\% & 13.31\% & 69.32\% \\ 
    SVM  & 57.40\% & 54.86\% & 56.26\% & 60.59\% & 56.45\% & 30.95\% & 35.81\% & 69.93\%   \\
    LSTM & 55.29\% & 52.32\% & 57.51\% & 55.90\% & 55.44\% & 28.96\% & 44.21\% & 69.18\% \\
    LSTM with \\ pre-trained word embedding    & \textbf{59.89\%} & \textbf{55.76\%} & \textbf{60.15\%} & 62.79\% & \textbf{59.23\%}  & \textbf{37.60\%} & \textbf{45.36\%} & \textbf{72.16\%} \\
    \bottomrule
  \end{tabular}
  \label{tab:table2}
\end{table}
***/
