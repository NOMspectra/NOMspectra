# masslib

small lib for working with FTCIR MS spectra

This lib supports
<ul>
    <li> Isotope Distribution generator </li>
    <li> Assigning brutto formulae to signal </li>
    <li> Working with spectra as with sets (intersection, union, etc) </li>
</ul>

Usage example for isotope distribution generator:
<pre>
    from masslib.distribution_generation.mass_distribution import IsotopeDistribution
    
    # brutto formulae that we want to use
    brutto = {"Pd": 1, "Cl": 2}
    
    # instance initialization
    d = IsotopeDistribution(brutto)
    
    # masses generations
    d.generate_iterations(100000)
    
    # plotting obtained distribution
    d.draw()
    
    # Graph can be saved by plt.savefig(filename, "png")
    # For that import matplotlib.pyplot as plt is needed
</pre>