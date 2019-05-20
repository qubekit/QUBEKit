# QUBEKit Style Guide for Development

## Introduction

The aim of QUBEKit is to produce easily accessible, current software for computational chemists.
QUBEKit pulls together the best methods in molecular dynamics, allowing in-depth analysis of important parameters.
QUBEKit is written collaboratively and as a result needs to be readable, modular and maintainable.
Our design philosophy is therefore focused towards encouraging stable, user-friendly code.

Generally, the style used in the QUBEKit code is consistent with [PEP8](https://www.python.org/dev/peps/pep-0008/).
This ensures a consistent, readable and pythonic style, easing development for multiple people.
As such, it is recommended that code is written using a fully-fledged IDE such as [PyCharm](https://www.jetbrains.com/pycharm/).
These IDEs often come with prebuilt style guides and help significantly with error reduction.


####Style Specifics

Beyond the usual style expectations of indenting, naming and so on, there are some key points that can be followed.

Think about your data structures:
* Could you use a set, tuple or dictionary instead of a list?
* Should you use a numpy array rather than a list?

Consider how the program will be used:
* Does this work for different-sized molecules?
* Is the output in an easily readable format?
* Are you taking advantage of parallelisation?
* Are there edge cases that might cause this to break?

Speed-ups:
* Is anything being run more often than it needs to be? Reloading configs, recalculating values etc.
* Would looping backwards help?
* If you're being exact, do you need to be? Is there a faster, approximate algorithm?
* Are there any unnecessary checks?
* How is the data being written to memory?
* Do you need to keep a file open for that long?

Readability:
* Are you following PEP8?
* Is the order of your comparison expression terse and logical?
* How is your spacing?
* Should you separate this section of code into a new function?
* Have you commented this section so that a similarly skilled programmer will quickly understand it?
* Is your object naming logical and clear?
* Have you written a docstring?

Maintainability:
* Are there edge cases which might cause issues?
* Have you tested multiple scenarios?
* Will this integrate with future developments?
* Is the documentation clear?
* Have you minimised dependencies?
* Is this modular?

Developers are encouraged to separate code into "paragraphs" of logic.
For example, loops and variable declarations should be surrounded by empty lines to organise the code:

    # Bad
    
    start = 0
    for count, line in enumerate(input_file):
        if search_word in line:
            start = count
    return count
    
    # Good
    
    start = 0
    
    for count, line in enumerate(input_file):
        if search_word in line:
            start = count
    
    return count

This is especially important for highly-nested loops. "Paragraphing" should be used to separate
according to each "thought" or segment. Partitioning like this significantly improves readability.

Following PEP8, continuation over new lines should be achieved using parentheses, rather than backslashes.

Imports should be as specific as is reasonable, to avoid excessive module loading.
If possible, avoid using imports altogether. Never use `import *` (even if it's what another package recommends).
Longer import names should be abbreviated. A common example is `import numpy as np`.

    # Terrible
    from math import *
    
    ans = factorial(3) // factorial(2)
    
    # Bad
    import math
    
    ans = math.factorial(3) // math.factorial(2)
    
    # Better
    from math import factorial as fact
    
    ans = fact(3) // fact(2)
    
    # Best
    ans = 3
    
If writing a new function or class, a docstring should be given describing its core purpose, including its inputs and outputs (if not obvious).
Comments should also be used to explain obscure or confusing code, or, ideally, rewrite the code to be more readable.

All functions should be independently testable. If a function requires a specific input file to run, it should be tested on 
files of varying size and complexity. Functions will not be merged until proven to be robust.

If a function or class is carrying out something complex or slow, append a statement to the log so the user understands something is happening.

Developers are strongly encouraged to use the "DRY" principle when coding.

Object naming examples:
    
    # Modules and files
    module.py
    other_module.py
    some_file.csv
    import module
    from module import SomeClass, some_function

    # Classes
    class ClassName(arg_1):
    class OtherClass:
    
    # Functions
    def function_name(arg_1, arg_2=0):
    
    # Variables
    var_name = True
    
If variable type is important and not obvious, a comment should be used to avoid confusion.
(Don't use the type hints)
    
    # returns a 2d numpy array
    return hessian
    
Using single underscores for private variables is optional but not encouraged.

Instead of using trailing underscores or misspellings to distinguish from built-in names,
change the object's name:

    # Bad
    indices = [...]
    for indx in indices:
    
    # Also bad
    indices = [...]
    for index_ in indices:
    
    # Good
    charges = [...]
    for charge in charges:
    
Avoid using short, ambiguous variable names. This is occasionally acceptable (dummy variables for example):

    # Bad
    atoms = [...]
    for i in atoms:
        print(i)
    
    # Good
    atoms = [...]
    for atom in atoms:
        print(atom)
    
    # OK
    squares = [x ** 2 for x in range(5)]

File names should be as specific as is reasonable with comment explanations where necessary:

    # Bad
    with open('example_file_name.dat', 'a+') as file:
    
    # Better
    with open('example_file_name.dat', 'a+') as input_file:

    # Best
    
    # input_file to be used by psi4:
    with open('example_file_name.dat', 'a+') as input_file:
    
Use context managers when it makes sense to do so.
If you are opening a file to read/write to, you are always going to want to close it again.
A context manager will do this for you, eliminating excess messy code, and reducing mistakes.

    # Bad
    file = open('example_file_name.dat', 'r')
    for line in file:
        ...
    file.close()
    
    # Good
    with open('example_file_name.dat', 'r') as file:
        for line in file:
            ...

In general, list comprehensions are preferred over the map, filter and reduce functions.
It should be obvious what a piece of code is doing, the above functions can be obfuscatory.
Of course, if the reverse is true, it is fine to use them.

Don't cram too much into one generator expression. sacrificing a little speed is well worth removing confusing code.

    # Bad
    nums = [i * j * k for i in range(5) for j in range(5) for k in range(5) if i * j * k % 2 == 0 and i * j < 12]

In the same vein, lambda functions and ternary operators should be used sparingly for readability.

Avoid bare exceptions.
Be explicit about what exception you're expecting and the error that should be raised as a result.

    # Bad
    try:
        import bonds
        
        bonds.perform_function(args)
        
    except:
        print('Could not import bonds.')

With the above bare except, you do not know if the script failed to import bonds, or if it failed for some other reason.

    # Better
    try:
        import bonds
        
        bonds.perform_function(args)
    
    except ImportError:
        raise ImportError('Could not import bonds.')

Else and finally clauses should also be used where appropriate.


#### Style Exceptions

As mentioned, PEP8 is generally followed, however there are some instances where deviation makes sense.
Also, when PEP8 is not specific, the implementations used in QUBEKit are here.

Due to the formatting of the input/output and job files, it is often appropriate to extend beyond the recommended line length. 
Sometimes, splitting a line can be less readable than simply running a few characters over, particularly in highly indented sections of code.
For example:

    subprocess.run(f'geometric-optimize --psi4 {self.molecule.name}.psi4in --nt {self.molecule.threads}', 
                   shell=True, stdout=log)

Splitting the string in this line would add confusion as to which arguments are parsed where and how the string formatting is carried out.
This is also relevant when dealing with highly nested sections of code. We have no hard limit on line length.

All strings should be written with single quotes. When both are used, double quotes should be inside the single quotes:

    # Here's a string:
    str_a = 'hello'
    
    # Here's a print statement with both quotes:
    print('Name given: "{name}"'.format(name='Chris'))


#### Closing Statement

QUBEKit is written with maintainability, collaboration and expansion in mind. 
As such, it needs to be easily read and understood by like-minded programmers.
If it hasn't been explicitly specified, try to use common sense to determine whether what has been written is appropriate.
