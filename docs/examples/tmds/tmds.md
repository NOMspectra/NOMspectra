# Assign by TMDS 

TMDS (Total mass difference statistics) is method for assign mass signal by brutto formulas. Signals are taken as a basis for which we can reliably determine and confirm the elemental composition, for example, those that have the corresponding peak of the C13 isotope. Then all the mass differences between all reliable signals are found and those that occur most often are selected. Then, the found mass differences are added to the masses of reliable signals and the peaks in the obtained region are searched for; if they match, they are assigned the elemental mass of the base peak + mass difference.

Read more here: Kunenkov, Erast V., et al. "Total mass difference statistics algorithm: a new approach to identification of high-mass building blocks in electrospray ionization Fourier transform ion cyclotron mass spectrometry data of natural organic matter." Analytical chemistry 81.24 (2009): 10106-10115.


```python
from nomspectra.spectrum import Spectrum
from nomspectra.diff import Tmds, assign_by_tmds
import nomspectra.draw as draw
```

Assign with minimal error - 0.25 for more reliable results

Show initial vk


```python
spec = Spectrum.read_csv('data/sample2.txt', mapper={'mass':'mass','intensity':'intensity'}, sep=',', take_columns=['mass','intensity'])
spec = spec.assign(brutto_dict={'C':(1,40),'H':(0,80), 'O':(0,40),'N':(0,2)}, rel_error=0.25)
draw.vk(spec)
```


    
![png](output_3_0.png)
    


## Caclculate TMDS

This may take quite a long time


```python
tmds_spec = Tmds(spec=spec).calc(p=0.2) #by varifiy p-value we can choose how much mass-diff we will take
tmds_spec = tmds_spec.assign(brutto_dict={'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(0,1)})
tmds_spec = tmds_spec.calc_mass()
tmds_spec.table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mass</th>
      <th>intensity</th>
      <th>C</th>
      <th>H</th>
      <th>O</th>
      <th>N</th>
      <th>assign</th>
      <th>calc_mass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.036</td>
      <td>1.264341</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>0.036385</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.016</td>
      <td>2.827907</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>2.015650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.995</td>
      <td>2.691473</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>3.994915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.068</td>
      <td>1.674419</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>4.067685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.011</td>
      <td>2.021705</td>
      <td>-1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>6.010565</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>322</th>
      <td>282.074</td>
      <td>0.504651</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>282.073955</td>
    </tr>
    <tr>
      <th>323</th>
      <td>284.053</td>
      <td>0.330233</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>284.053220</td>
    </tr>
    <tr>
      <th>324</th>
      <td>286.105</td>
      <td>0.467442</td>
      <td>13.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>286.105255</td>
    </tr>
    <tr>
      <th>325</th>
      <td>294.074</td>
      <td>0.395349</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>294.073955</td>
    </tr>
    <tr>
      <th>326</th>
      <td>296.053</td>
      <td>0.357364</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>296.053220</td>
    </tr>
  </tbody>
</table>
<p>327 rows × 8 columns</p>
</div>



We can plot total mass diference spectrum


```python
draw.spectrum(tmds_spec)
```


    
![png](output_7_0.png)
    


## Assigne by TMDS

It is takes a lot of times


```python
spec = assign_by_tmds(spec, tmds_spec, rel_error=3)
#show percent of process complete
```

    100%|██████████| 327/327 [02:02<00:00,  2.67it/s]


Look result


```python
draw.vk(spec)
```


    
![png](output_11_0.png)
    


Look how it will be Without TMDS and rel_error 0.5 ppm


```python
spec = Spectrum.read_csv('data/sample2.txt', mapper={'mass':'mass','intensity':'intensity'}, sep=',', take_columns=['mass','intensity'])
spec = spec.assign(brutto_dict={'C':(1,40),'H':(0,80), 'O':(0,40),'N':(0,3)}, rel_error=0.5)
draw.vk(spec)
```


    
![png](output_13_0.png)
    


## TMDS setting

take 100 diff-masses from tmds

If you want to acselerate treatment or restrict tmds by number you can restrict it. But for best result it is better to use 400-500 mass-diffrences from tmds.


```python
spec = Spectrum.read_csv('data/sample2.txt', mapper={'mass':'mass','intensity':'intensity'}, sep=',', take_columns=['mass','intensity'])
spec = spec.assign(brutto_dict={'C':(1,40),'H':(0,80), 'O':(0,40),'N':(0,3)}, rel_error=0.25)

spec = assign_by_tmds(spec, max_num=100)

draw.vk(spec)
```

    100%|██████████| 101/101 [00:40<00:00,  2.51it/s]



    
![png](output_15_1.png)
    


take tmds with p-value = 0.7

Some times with default p-value = 0.2 tmds spectrum will be too big, so its reasonobly restrict it by p-value.


```python
spec = Spectrum.read_csv('data/sample2.txt', mapper={'mass':'mass','intensity':'intensity'}, sep=',', take_columns=['mass','intensity'])
spec = spec.assign(brutto_dict={'C':(1,40),'H':(0,80), 'O':(0,40),'N':(0,3)}, rel_error=0.25)

spec = assign_by_tmds(spec, p=0.7)

draw.vk(spec)
```

    100%|██████████| 149/149 [00:58<00:00,  2.54it/s]



    
![png](output_17_1.png)
    


take tmds with p-value = 1.0 but without verification by C13_peaks

Actually it is better use C13 validation, but when spectrum consist only 500-1000 reliable ions, it may be good desicion because tmds may be toot small.


```python
spec = Spectrum.read_csv('data/sample2.txt', mapper={'mass':'mass','intensity':'intensity'}, sep=',', take_columns=['mass','intensity'])
spec = spec.assign(brutto_dict={'C':(1,40),'H':(0,80), 'O':(0,40),'N':(0,3)}, rel_error=0.25)

spec = assign_by_tmds(spec, p=1.0, C13_filter=False)

draw.vk(spec)
```

    100%|██████████| 37/37 [00:15<00:00,  2.39it/s]



    
![png](output_19_1.png)
    

