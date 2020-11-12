###################################################
#
#  Script for producing line input files for joebvp
#
#  Quick usage: linesTogether.processFile([paramfile])
###################################################

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
from astropy.io import ascii
from joebvp.joebgoodies import closest
import pdb

buf = 5  #number of pixels between pix2 and pix1 for lines to be unblended

def picksyslines(z,comps):
	''' Return all lines at a systemic redshift
	'''
	zval=comps['z'][closest(comps['z'],z)]
	zmatches=np.where(np.abs(comps['z']-zval)<0.0001)[0]
	return [comps['idxs'][idx] for idx in zmatches]

def pickspecieslines(trans,comps):
    ''' Return the components of a given species and the lines in these components.  
    '''
    transmatches=np.where(comps['trans']==trans)[0]
    return transmatches,[comps['idxs'][idx] for idx in transmatches]

def gettrans(lineidx,comps):
	''' Once groups of trans have been identified, find which trans group a line belongs to.
    '''
	for i in range(len(comps['idxs'])):
		cpidx=[]
		try:
			# Return index of line within trans list (idx not actually used, just a check)
			idx=comps['idxs'][i].index(lineidx) 
			cpidx=i
			break
		except:
			pass
	return cpidx

def getblended(lineidx,blends):
    ''' Once blends have been identified, find which blend a given line index belongs to.
    '''
    for i in range(len(blends)):
        blidx=[]
        try:
            # Return index of line within blend list (idx not actually used, just a check)
            idx=blends[i].index(lineidx) 
            blidx=i
            break
        except:
            pass
    return blidx

def findblends(linepars,pixbuffer=10):
    ''' Operates on full line parameter list to group all blends.
    '''
    blends = []
    for i in range(len(linepars)-1):
        if i == 0: 
            thisblend = [i]
        print(thisblend)
        if (linepars['pix1'][i+1] - linepars['pix2'][i]) < pixbuffer: 
            thisblend.append(i+1)
            if i==(len(linepars)-2):
            	blends.append(thisblend)
        else:
            blends.append(thisblend)
            thisblend = [i+1]
            if i==(len(linepars)-2):
            	blends.append(thisblend)
    return blends

def compilecomps(linepars):
    ''' Operates on full line parameter list to group all transitions belonging to a
    specific component (species and redshift).
    
    Output:  Dictionary that includes species name, component redshift, and indices of 
    lines belonging to each component.
    '''
    uqzs=np.unique(linepars['zsys'])
    comps=[]
    compzs=[]
    comptrans=[]
    for idx,zz in enumerate(uqzs):
        thisz=np.where(linepars['zsys']==zz)[0]
        uqtrans=np.unique(linepars['trans'][thisz])
        for j,tt in enumerate(uqtrans):
                sametrans=np.where((linepars['zsys']==zz)&(linepars['trans']==tt))[0]
                comps.append(sametrans.tolist())
                compzs.append(zz)
                comptrans.append(tt)
    compdict = {'z': np.array(compzs), 'trans':np.array(comptrans), 'idxs':np.array(comps)}
    return compdict
            
def compilelist_z(z,comps,blends):
	''' Put together list of lines from an initial desired redshift
	Note:  Does not enter the rabbit hole and does an incomplete job
	'''
	idxs=picksyslines(z,comps)
	idxs=np.concatenate(idxs)
	compilation=[]
	for i,idx in enumerate(idxs):
		bl=getblended(idx,blends)
		if bl!=None: compilation.append(np.array(bl))
	conc=np.concatenate(compilation)
	return np.unique(conc).tolist()

def compilelist_trans(trans,comps,blends):
	''' Put together list of lines from an initial desired transition
	Note:  Does not enter the rabbit hole and does an incomplete job
	'''
	blfound=[] ; cpfound=[]
	cpidxs,idxs=picktranslines(trans,comps)
	cpfound.append(cpidxs)
	idxs=np.concatenate(idxs)
	compilation=[]
	for i,idx in enumerate(idxs):
		bl=getblended(idx,blends)
		if bl!=None: compilation.append(np.array(bl))
	conc=np.concatenate(compilation)
	return np.unique(conc).tolist()

def rabbithole(lineidx,comps,blends):
	''' Given a line index, find all blended lines, transitions of those lines, blends
	of those transitions...
	'''
	blfound=[] ; cpfound=[]  # These will hold indices into  the 'blends' and 'comps' LoLs
	newbl2try=[] ; newcp2try=[]  # Each iteration of while loop will find new blends and trans
	bl=getblended(lineidx,blends)  # See if line is blended
	print(bl)
	if len(blends[bl])>1:
		blfound.append(bl)  # Initialize with blends of input line
		newbl2try=blfound
	else:
		cpidx=gettrans(lineidx,comps)
		cpfound.append(cpidx)
		newcp2try=cpfound
		#return np.array([lineidx])  # If no blends, just return line
		
	newbl2try=blfound  # 1st trip through the loop will find transitions of the blend group
	while (len(newbl2try)>0)|(len(newcp2try)>0):
		newbl=newbl2try ; newcp=newcp2try  # Store indices to try
		newbl2try=[] ; newcp2try=[]	 # Make way for the new indices
		if len(newbl)>0:  # Find all transitions of lines in new blend group found last time
			for nb in newbl:  
				print(nb)
				for idx in blends[nb]:
					cpidx=gettrans(idx,comps)
					if cpidx not in cpfound:  # Make sure trans group is new
						cpfound.append(cpidx)  # Store trans group
						newcp2try.append(cpidx)  # Save trans group to look for blends
		if len(newcp)>0:  # Find the blends of lines in new transition group found last time 
			for nc in newcp:
				for idx in comps['idxs'][nc]:
					blidx=getblended(idx,blends)
					if blidx not in blfound:  # Make sure blend group is new
						blfound.append(blidx)  # Store blend group
						newbl2try.append(blidx)  # Save blend group to look for other trans
	blarr=np.array(blends) ; cparr=np.array(comps['idxs'])
	blarr=np.concatenate(blarr[blfound])
	cparr=np.concatenate(cparr[cpfound])
	idxs2fit=np.union1d(blarr,cparr)
	return idxs2fit
    
def findfitgroups(linepars,comps,blends):
	fgroups=[]
	idxlist=range(len(linepars))
	for idx in idxlist:
		print(idx)
		thisg=rabbithole(idx,comps,blends)
		for tgidx in thisg:
			try:
				idxlist.remove(tgidx)
			except:
				break
		fgroups.append(thisg)
	return fgroups
	
	

def writeparfile(linepars,idx,filename):
    #outlinefile=open(filename,'wb')
    #outlinefile.write('specfile|restwave|zsys|col|bval|vel|nflag|bflag|vflag|vlim1|vlim2|wobs1|wobs2|pix1|pix2|trans \n')
    ascii.write(linepars[idx],output=filename,delimiter='|')

    
def initfile(filename,wave1=None,wave2=None,**kwargs):
	linepars=ascii.read(filename)
	linepars.sort('pix1')
	if wave1==None:
		wave1=linepars['wobs1'][0]
	if wave2==None:
		wave2=linepars['wobs2'][-1]
	linepars=linepars[(linepars['wobs1']>=wave1)&(linepars['wobs2']<=wave2)]
	blends=findblends(linepars,**kwargs)
	comps=compilecomps(linepars)
	return linepars,comps,blends
	
def processfile(filename,outputprefix='vpinput_',smallgroup=1,**kwargs):
	linepars,comps,blends=initfile(filename,**kwargs)
	fgs=findfitgroups(linepars,comps,blends)
	fgs=np.array(fgs)
	lengths=np.zeros(len(fgs))
	for i,gr in enumerate(fgs):
		lengths[i]=len(gr)
	smallies=np.where(lengths<=smallgroup)[0]
	allsms=np.concatenate(fgs[smallies])
	writeparfile(linepars,allsms,outputprefix+'0.dat')
	fgs=np.delete(fgs,smallies)
	for i,gr in enumerate(fgs):
		writeparfile(linepars,gr,outputprefix+str(i+1)+'.dat')
	
def findunblended(filename,outputfile='unblended.dat',**kwargs):
	linepars,comps,blends=initfile(filename,**kwargs)
	unbl=[bl[0] for bl in blends if len(bl)==1]
	writeparfile(linepars,unbl,outputfile)
	


filename='pg1407testsyslines.dat'

