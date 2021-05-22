VischeckJ
version 1.1
November 23, 2006
(this file last updated November 23, 2006)
Unlike version 1.0, version 1.1 uses the same display parameter assumptions
as the online version of Vischeck. Therefore, the simulation results should
now be in better agreement with the online simulation.

QUICKSTART

1. Install ImageJ (http://rsb.info.nih.gov/ij/)
2. Dowload VischeckJ from http://vischeck.com
3. Unzip VischeckJ1.zip
4. Move 'Vischeck_Panel.class' and 'Vischeck.class' to .../ImageJ/plugins/
5. Start ImageJ


DETAILED INSTALLATION INSTRUCTIONS

VischeckJ is a plug-in for the ImageJ public domain image analysis package
from the US National Institutes of Health.  ImageJ is java-based, so it 
should run on any operating system that has a Java virtural machine (JVM). 
This includes Windows, Mac and Linux.

You will need to install ImageJ before you can use VischeckJ. ImageJ can be 
downloaded from the website:

   http://rsb.info.nih.gov/ij/

ImageJ will install itself into a folder. You can choose the name of this folder
if you like - the default is 'ImageJ'. Inside this folder you will find the
ImageJ executable, some other files and a folder called 'plugins'. This is 
where you will put VischeckJ.

Once you have installed ImageJ download VischeckJ. VischeckJ comes as a 
single compressed '.zip' file. You can unzip this on any machine.

  - On Windows use a program like 'WinZip'.
  - On a Mac, use a newish version of 'Stuffit'.
  - On linux, type 'unzip vischeckJ_1.zip'.

This should create a folder containing three files: ReadMe.txt (this file) and
the two plug-in files: Vischeck_Panel.class and Vischeck.class.

Simply place the Vischeck_Panel.class and Vischeck.class into your ImageJ 
"plugins" folder. Start ImageJ (or re-start it if it was already running), 
and you should find the "Vischeck Panel" option under ImageJ's 'Plugins' menu.


USAGE TIPS

Linux: Note that ImageJ does not seem to work well on the Kaffe JVM that 
comes standard on many Linux distributions (such as RedHat).  You may want 
to update your JVM or use the JRE 1.1.8 JVM that now comes with ImageJ 
when you use the nifty InstallAnywhere installer.  Even if you already have 
a JVM installed on your system, you may consider installing ImageJ this way.  
You will end up with an extra JVM, but not all JVMs are created equal, and 
ImageJ works particularly well on JRE 1.1.8.  In fact, you can run it under 
different JVMs and see the performance differences. 


ABOUT VISCHECK

Vischeck draws on algorithms developed at many different vision 
laboratories around the world. In particular, this plug-in owes a 
great deal to a 1997 paper by Brettel, Vienot and Mollon entitled 
"Computerized simulation of color appearance for dichromats" 
(Journal of the Optical Society of America v14 pp2647) and
to recent work from Brian Wandell's group at Stanford
(http://white.stanford.edu).

VischeckJ is Copyright 2001 by Tiny Eyes, Inc.  It was written
by Bob Dougherty (bobd@stanford.edu) using Sun's JDK 1.3 for Linux.
It is based on code written by Alex Wade (alexwade@stanford.edu) 
and Bob Dougherty.

VischeckJ is freely available for use and distribution.  You have 
our permission to include VischeckJ on a CD-ROM, or any other media, 
provided it is sold at a minimal price to cover production and 
promotional costs only.


DISCLAIMER
"VischeckJ" IS SUPPLIED AS IS. THE AUTHORS DISCLAIM
ALL WARRANTIES, EXPRESSED OR IMPLIED, INCLUDING,
WITHOUT LIMITATION, THE WARRANTIES OF
MERCHANTABILITY AND OF FITNESS FOR ANY PURPOSE.
THE AUTHORS ASSUME NO LIABILITY FOR DAMAGES,
DIRECT OR CONSEQUENTIAL, WHICH MAY RESULT FROM
THE USE OF "VischeckJ".
