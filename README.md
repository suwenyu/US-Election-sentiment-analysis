# Twitter Sentiment Analysis of the 2016 U.S. Presidential Election

This project proposed to use different models with several pre-processing methods for sentiment prediction. Experiments conducted on two datasets which are Obama and Romney attest the effectiveness of different models.

### Result

##### Table 1: Effect of preprocessing

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky" colspan="2">Obama</th>
    <th class="tg-0pky" colspan="2">Romney</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fn5d">Model</td>
    <td class="tg-fn5d">NB</td>
    <td class="tg-fn5d">SVM</td>
    <td class="tg-fn5d">NB</td>
    <td class="tg-fn5d">SVM</td>
  </tr>
  <tr>
    <td class="tg-fn5d">Removing punctuation, stopwords and doing lowercase</td>
    <td class="tg-fn5d">55.59%</td>
    <td class="tg-fn5d">55.02%</td>
    <td class="tg-fn5d">53.01%</td>
    <td class="tg-fn5d">53.15%</td>
  </tr>
  <tr>
    <td class="tg-exjq">method 1 and removing HTML tags and URLs</td>
    <td class="tg-7k6d">57.60%</td>
    <td class="tg-exjq">57.37%</td>
    <td class="tg-exjq">53.74%</td>
    <td class="tg-exjq">56.27%</td>
  </tr>
  <tr>
    <td class="tg-0lax">method 1, 2 and deconstruct</td>
    <td class="tg-0lax">57.33%</td>
    <td class="tg-0lax">57.40%</td>
    <td class="tg-1wig">53.86%</td>
    <td class="tg-1wig">56.45%</td>
  </tr>
  <tr>
    <td class="tg-0lax">method 1,2 and correct misspelling</td>
    <td class="tg-0lax">57.54%</td>
    <td class="tg-1wig">57.47%</td>
    <td class="tg-0lax">53.74%</td>
    <td class="tg-0lax">56.41%</td>
  </tr>
</tbody>
</table>

##### Table 2: Pre-process for deep learning
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">Coverage of vocabulary</th>
    <th class="tg-0pky">Coverage of Text</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">None</td>
    <td class="tg-0pky">38.14%</td>
    <td class="tg-0pky">68.88%</td>
  </tr>
  <tr>
    <td class="tg-0pky">removing punctuation</td>
    <td class="tg-0pky">58.93%</td>
    <td class="tg-0pky">82.48%</td>
  </tr>
  <tr>
    <td class="tg-0pky">correct misspelling</td>
    <td class="tg-0pky">58.92%</td>
    <td class="tg-0pky">80.89%</td>
  </tr>
</tbody>
</table>

##### Table 3: Effect of deep learning
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Accuracy</th>
    <th class="tg-0lax">Obama</th>
    <th class="tg-0lax">Romney</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Naive Bayes</td>
    <td class="tg-0lax">57.54%</td>
    <td class="tg-0lax">53.74%</td>
  </tr>
  <tr>
    <td class="tg-0lax">SVM</td>
    <td class="tg-0lax">57.40%</td>
    <td class="tg-0lax">56.45%</td>
  </tr>
  <tr>
    <td class="tg-0lax">LSTM</td>
    <td class="tg-0lax">55.29%</td>
    <td class="tg-0lax">55.44%</td>
  </tr>
  <tr>
    <td class="tg-0lax">LSTM w. pre-trained Model</td>
    <td class="tg-1wig">59.89%</td>
    <td class="tg-1wig">59.23%</td>
  </tr>
</tbody>
</table>
