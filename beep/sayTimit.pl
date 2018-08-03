#! /usr/local/bin/perl

# sayTimit

# Paul Callaghan, may1994.
# University of Durham.

# instructions elsewhere (in file sayTimit.doc)

%timit2ota = (
"b", "b",     
"d", "d",     
"g", "g",     
"p", "p",     
"t", "t",     
"k", "k",     
"dx", "d",    
"q", "t",     


"jh", "dZ",    
"ch", "tS",    


"s", "s",     
"sh", "S",    
"z", "z",     
"zh", "Z",    
"f", "f",     
"th", "T",    
"v", "v",     
"dh", "D",    


"m", "m",     
"n", "n",     
"ng", "N",    
"em", "m",    
"en", "n",    
"eng", "N",   
"nx", "n",    



"l", "l",     
"r", "r",     
"w", "w",     
"y", "j",     
"hh", "h",    
"hv", "h",    
"el", "l",    




"iy", "i",    
"ih", "I",    
"eh", "e",    

"ea", "e@",				# eg bare, or air. 
"ey", "eI",    
"ae", "&",    
"aa", "A",    
"aw", "aU",    
"ay", "aI",    
"ah", "V",    

"oh", "0",    
"oy", "oI",    
"ow", "@U",    
"uh", "U",    
"uw", "u",    
"ux", "u",    

"er", "3",    
"ax", "@",    
"ix", "I",    
"axr", "R",   
"ax-h", "e",				# forgotten what this is!
					# non-timit symbols for RP Vowels
"ia", "I@",					# as in 'beer'
"ao", "O",					# as in 'cord'
"ua", "U@",					# as in 'tour'

"epi", " ",					# epenthetic silence
"sil", " ",					# silence
"pau", " ",					# pause

"1", "'",					# primary stress
"2", ","					# secondary stress.
	  );

################################################################################
# crank up 'say'.

# "say:sound <params>" needs to be changed to your installation of
# rsynth, with desired parameters.

open(SAY, "| say:sound +h -g 0.2") || die "Couldn't start SAY: $!\n";	

# then make it flush after every write/print operation. The method is to
# set it temporarily as the default output channel, then get it to flush,
# then reset the old default channel. Surprisingly (for some ppl), SAY will
# STILL flush as required. 

$oldofh = select(SAY);				# make it default 
$| = 1;						# flush after each IO op.
select($oldofh);				# and reset.


# open input file.

open(INPUT, $ARGV[0]) || die "Couldn't open pronunciations file $ARGV[0].\n";
shift;

# startfrom word? 
$startfrom = $ARGV[0];

if ($startfrom ne "") {
	do {
		$_ = <INPUT>;
		@tmp = split;
	} until ( $tmp[0] eq $startfrom);
	print "startfrom " . $_ . "\n";
		
} else {
	$_ = <INPUT>;		# first line expected 
}
	

# main loop.

START:
do {
	$orig = $_;
	s/#.*//;				# kill comment
	s/[.]//g;				# kill '.' 

	s/([a-z]*)1/1 \1/g;
	s/([a-z]*)2/2 \1/g;			
				# change stress notation: OTA seems to require
				# marks BEFORE the syllable, not AFTER the
				# 'nuclear'(?) vowel.

	@tmp = split;

	$phons = "";
	foreach $p (@tmp[1..$#tmp]) {
		if ($timit2ota{$p} eq "") {
			print "\nERROR: unknown symbol " . $p;
		} else {
			$phons .= $timit2ota{$p};
		}
	}
	$tmp[0] =~ tr/A-Z/a-z/;		# current word to lower case.

	$ig = 0;
	do {
		print STDOUT ">> " . $orig;
		print STDOUT "== " . $phons . "\n";
		unless ($ig) { print SAY "[" . $phons . "]\n"; }
						# write result to SAY
		$ig = 0;
		$cmd = <STDIN>;
		chop $cmd;
		if ($cmd =~ /^s/) { $cmd =~ s/^s//; if ($cmd eq "") { print SAY $tmp[0] . "\n"; } else { print SAY $cmd . "\n"; } $ig = 1;}
		if ($cmd =~ /^t /) { $old = $_; $cmd =~ s/^t //; $_ = $cmd; goto START; }
		if ($cmd =~ /^p/) { $_ = $old; goto START; }

	} until ($cmd eq "n");
} while (<INPUT>);	

close(INPUT);
close(SAY);
