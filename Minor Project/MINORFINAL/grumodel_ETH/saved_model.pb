û9
í¾
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¨ë6
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:d*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

gru_6/gru_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*(
shared_namegru_6/gru_cell_6/kernel

+gru_6/gru_cell_6/kernel/Read/ReadVariableOpReadVariableOpgru_6/gru_cell_6/kernel*
_output_shapes
:	¬*
dtype0

!gru_6/gru_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*2
shared_name#!gru_6/gru_cell_6/recurrent_kernel

5gru_6/gru_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_6/gru_cell_6/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_6/gru_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*&
shared_namegru_6/gru_cell_6/bias

)gru_6/gru_cell_6/bias/Read/ReadVariableOpReadVariableOpgru_6/gru_cell_6/bias*
_output_shapes
:	¬*
dtype0

gru_7/gru_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*(
shared_namegru_7/gru_cell_7/kernel

+gru_7/gru_cell_7/kernel/Read/ReadVariableOpReadVariableOpgru_7/gru_cell_7/kernel*
_output_shapes
:	d¬*
dtype0

!gru_7/gru_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*2
shared_name#!gru_7/gru_cell_7/recurrent_kernel

5gru_7/gru_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_7/gru_cell_7/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_7/gru_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*&
shared_namegru_7/gru_cell_7/bias

)gru_7/gru_cell_7/bias/Read/ReadVariableOpReadVariableOpgru_7/gru_cell_7/bias*
_output_shapes
:	¬*
dtype0

gru_8/gru_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*(
shared_namegru_8/gru_cell_8/kernel

+gru_8/gru_cell_8/kernel/Read/ReadVariableOpReadVariableOpgru_8/gru_cell_8/kernel*
_output_shapes
:	d¬*
dtype0

!gru_8/gru_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*2
shared_name#!gru_8/gru_cell_8/recurrent_kernel

5gru_8/gru_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_8/gru_cell_8/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_8/gru_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*&
shared_namegru_8/gru_cell_8/bias

)gru_8/gru_cell_8/bias/Read/ReadVariableOpReadVariableOpgru_8/gru_cell_8/bias*
_output_shapes
:	¬*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/gru_6/gru_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*/
shared_name Adam/gru_6/gru_cell_6/kernel/m

2Adam/gru_6/gru_cell_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_6/gru_cell_6/kernel/m*
_output_shapes
:	¬*
dtype0
­
(Adam/gru_6/gru_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_6/gru_cell_6/recurrent_kernel/m
¦
<Adam/gru_6/gru_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_6/gru_cell_6/recurrent_kernel/m*
_output_shapes
:	d¬*
dtype0

Adam/gru_6/gru_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_6/gru_cell_6/bias/m

0Adam/gru_6/gru_cell_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_6/gru_cell_6/bias/m*
_output_shapes
:	¬*
dtype0

Adam/gru_7/gru_cell_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*/
shared_name Adam/gru_7/gru_cell_7/kernel/m

2Adam/gru_7/gru_cell_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_7/kernel/m*
_output_shapes
:	d¬*
dtype0
­
(Adam/gru_7/gru_cell_7/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_7/gru_cell_7/recurrent_kernel/m
¦
<Adam/gru_7/gru_cell_7/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_7/gru_cell_7/recurrent_kernel/m*
_output_shapes
:	d¬*
dtype0

Adam/gru_7/gru_cell_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_7/gru_cell_7/bias/m

0Adam/gru_7/gru_cell_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_7/bias/m*
_output_shapes
:	¬*
dtype0

Adam/gru_8/gru_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*/
shared_name Adam/gru_8/gru_cell_8/kernel/m

2Adam/gru_8/gru_cell_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_8/gru_cell_8/kernel/m*
_output_shapes
:	d¬*
dtype0
­
(Adam/gru_8/gru_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_8/gru_cell_8/recurrent_kernel/m
¦
<Adam/gru_8/gru_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_8/gru_cell_8/recurrent_kernel/m*
_output_shapes
:	d¬*
dtype0

Adam/gru_8/gru_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_8/gru_cell_8/bias/m

0Adam/gru_8/gru_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_8/gru_cell_8/bias/m*
_output_shapes
:	¬*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/gru_6/gru_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*/
shared_name Adam/gru_6/gru_cell_6/kernel/v

2Adam/gru_6/gru_cell_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_6/gru_cell_6/kernel/v*
_output_shapes
:	¬*
dtype0
­
(Adam/gru_6/gru_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_6/gru_cell_6/recurrent_kernel/v
¦
<Adam/gru_6/gru_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_6/gru_cell_6/recurrent_kernel/v*
_output_shapes
:	d¬*
dtype0

Adam/gru_6/gru_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_6/gru_cell_6/bias/v

0Adam/gru_6/gru_cell_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_6/gru_cell_6/bias/v*
_output_shapes
:	¬*
dtype0

Adam/gru_7/gru_cell_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*/
shared_name Adam/gru_7/gru_cell_7/kernel/v

2Adam/gru_7/gru_cell_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_7/kernel/v*
_output_shapes
:	d¬*
dtype0
­
(Adam/gru_7/gru_cell_7/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_7/gru_cell_7/recurrent_kernel/v
¦
<Adam/gru_7/gru_cell_7/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_7/gru_cell_7/recurrent_kernel/v*
_output_shapes
:	d¬*
dtype0

Adam/gru_7/gru_cell_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_7/gru_cell_7/bias/v

0Adam/gru_7/gru_cell_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_7/gru_cell_7/bias/v*
_output_shapes
:	¬*
dtype0

Adam/gru_8/gru_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*/
shared_name Adam/gru_8/gru_cell_8/kernel/v

2Adam/gru_8/gru_cell_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_8/gru_cell_8/kernel/v*
_output_shapes
:	d¬*
dtype0
­
(Adam/gru_8/gru_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_8/gru_cell_8/recurrent_kernel/v
¦
<Adam/gru_8/gru_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_8/gru_cell_8/recurrent_kernel/v*
_output_shapes
:	d¬*
dtype0

Adam/gru_8/gru_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_8/gru_cell_8/bias/v

0Adam/gru_8/gru_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_8/gru_cell_8/bias/v*
_output_shapes
:	¬*
dtype0

NoOpNoOp
M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÑL
valueÇLBÄL B½L
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
Á
 cell
!
state_spec
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&_random_generator
'__call__
*(&call_and_return_all_conditional_losses*
¦

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
 
1iter

2beta_1

3beta_2
	4decay
5learning_rate)m*m6m7m8m9m:m;m<m=m>m)v*v6v7v8v9v:v;v<v=v>v*
R
60
71
82
93
:4
;5
<6
=7
>8
)9
*10*
R
60
71
82
93
:4
;5
<6
=7
>8
)9
*10*
* 
°
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Dserving_default* 
Ó

6kernel
7recurrent_kernel
8bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I_random_generator
J__call__
*K&call_and_return_all_conditional_losses*
* 

60
71
82*

60
71
82*
* 


Lstates
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
Ó

9kernel
:recurrent_kernel
;bias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses*
* 

90
:1
;2*

90
:1
;2*
* 


Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
Ó

<kernel
=recurrent_kernel
>bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c_random_generator
d__call__
*e&call_and_return_all_conditional_losses*
* 

<0
=1
>2*

<0
=1
>2*
* 


fstates
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
"	variables
#trainable_variables
$regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_6/gru_cell_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_6/gru_cell_6/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_6/gru_cell_6/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_7/gru_cell_7/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_7/gru_cell_7/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_7/gru_cell_7/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_8/gru_cell_8/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_8/gru_cell_8/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_8/gru_cell_8/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

q0*
* 
* 
* 

60
71
82*

60
71
82*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 

90
:1
;2*

90
:1
;2*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 

<0
=1
>2*

<0
=1
>2*
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

 0*
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_6/gru_cell_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_6/gru_cell_6/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_6/gru_cell_6/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_7/gru_cell_7/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_7/gru_cell_7/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_7/gru_cell_7/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_8/gru_cell_8/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_8/gru_cell_8/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_8/gru_cell_8/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_6/gru_cell_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_6/gru_cell_6/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_6/gru_cell_6/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_7/gru_cell_7/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_7/gru_cell_7/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_7/gru_cell_7/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_8/gru_cell_8/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_8/gru_cell_8/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_8/gru_cell_8/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_gru_6_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿd
â
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_6_inputgru_6/gru_cell_6/biasgru_6/gru_cell_6/kernel!gru_6/gru_cell_6/recurrent_kernelgru_7/gru_cell_7/biasgru_7/gru_cell_7/kernel!gru_7/gru_cell_7/recurrent_kernelgru_8/gru_cell_8/biasgru_8/gru_cell_8/kernel!gru_8/gru_cell_8/recurrent_kerneldense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_49173
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+gru_6/gru_cell_6/kernel/Read/ReadVariableOp5gru_6/gru_cell_6/recurrent_kernel/Read/ReadVariableOp)gru_6/gru_cell_6/bias/Read/ReadVariableOp+gru_7/gru_cell_7/kernel/Read/ReadVariableOp5gru_7/gru_cell_7/recurrent_kernel/Read/ReadVariableOp)gru_7/gru_cell_7/bias/Read/ReadVariableOp+gru_8/gru_cell_8/kernel/Read/ReadVariableOp5gru_8/gru_cell_8/recurrent_kernel/Read/ReadVariableOp)gru_8/gru_cell_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp2Adam/gru_6/gru_cell_6/kernel/m/Read/ReadVariableOp<Adam/gru_6/gru_cell_6/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_6/gru_cell_6/bias/m/Read/ReadVariableOp2Adam/gru_7/gru_cell_7/kernel/m/Read/ReadVariableOp<Adam/gru_7/gru_cell_7/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_7/gru_cell_7/bias/m/Read/ReadVariableOp2Adam/gru_8/gru_cell_8/kernel/m/Read/ReadVariableOp<Adam/gru_8/gru_cell_8/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_8/gru_cell_8/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp2Adam/gru_6/gru_cell_6/kernel/v/Read/ReadVariableOp<Adam/gru_6/gru_cell_6/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_6/gru_cell_6/bias/v/Read/ReadVariableOp2Adam/gru_7/gru_cell_7/kernel/v/Read/ReadVariableOp<Adam/gru_7/gru_cell_7/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_7/gru_cell_7/bias/v/Read/ReadVariableOp2Adam/gru_8/gru_cell_8/kernel/v/Read/ReadVariableOp<Adam/gru_8/gru_cell_8/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_8/gru_cell_8/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_53019
ï

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_6/gru_cell_6/kernel!gru_6/gru_cell_6/recurrent_kernelgru_6/gru_cell_6/biasgru_7/gru_cell_7/kernel!gru_7/gru_cell_7/recurrent_kernelgru_7/gru_cell_7/biasgru_8/gru_cell_8/kernel!gru_8/gru_cell_8/recurrent_kernelgru_8/gru_cell_8/biastotalcountAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/gru_6/gru_cell_6/kernel/m(Adam/gru_6/gru_cell_6/recurrent_kernel/mAdam/gru_6/gru_cell_6/bias/mAdam/gru_7/gru_cell_7/kernel/m(Adam/gru_7/gru_cell_7/recurrent_kernel/mAdam/gru_7/gru_cell_7/bias/mAdam/gru_8/gru_cell_8/kernel/m(Adam/gru_8/gru_cell_8/recurrent_kernel/mAdam/gru_8/gru_cell_8/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/gru_6/gru_cell_6/kernel/v(Adam/gru_6/gru_cell_6/recurrent_kernel/vAdam/gru_6/gru_cell_6/bias/vAdam/gru_7/gru_cell_7/kernel/v(Adam/gru_7/gru_cell_7/recurrent_kernel/vAdam/gru_7/gru_cell_7/bias/vAdam/gru_8/gru_cell_8/kernel/v(Adam/gru_8/gru_cell_8/recurrent_kernel/vAdam/gru_8/gru_cell_8/bias/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_53149ø£5
B
þ
while_body_49288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_6_readvariableop_resource_0:	¬D
1while_gru_cell_6_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_6_readvariableop_resource:	¬B
/while_gru_cell_6_matmul_readvariableop_resource:	¬D
1while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_6/MatMul/ReadVariableOp¢(while/gru_cell_6/MatMul_1/ReadVariableOp¢while/gru_cell_6/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_6/ReadVariableOpReadVariableOp*while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_6/unstackUnpack'while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_6/BiasAddBiasAdd!while/gru_cell_6/MatMul:product:0!while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_6/splitSplit)while/gru_cell_6/split/split_dim:output:0!while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_6/MatMul_1MatMulwhile_placeholder_20while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_6/BiasAdd_1BiasAdd#while/gru_cell_6/MatMul_1:product:0!while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_6/split_1SplitV#while/gru_cell_6/BiasAdd_1:output:0while/gru_cell_6/Const:output:0+while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_6/addAddV2while/gru_cell_6/split:output:0!while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_6/SigmoidSigmoidwhile/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_1AddV2while/gru_cell_6/split:output:1!while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_1Sigmoidwhile/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mulMulwhile/gru_cell_6/Sigmoid_1:y:0!while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_2AddV2while/gru_cell_6/split:output:2while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/mul_1Mulwhile/gru_cell_6/beta:output:0while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_2Sigmoidwhile/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_2Mulwhile/gru_cell_6/add_2:z:0while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/IdentityIdentitywhile/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_6/IdentityN	IdentityNwhile/gru_cell_6/mul_2:z:0while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49338*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_3Mulwhile/gru_cell_6/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/subSubwhile/gru_cell_6/sub/x:output:0while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_4Mulwhile/gru_cell_6/sub:z:0#while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_3AddV2while/gru_cell_6/mul_3:z:0while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_6/MatMul/ReadVariableOp)^while/gru_cell_6/MatMul_1/ReadVariableOp ^while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_6_matmul_1_readvariableop_resource3while_gru_cell_6_matmul_1_readvariableop_resource_0"d
/while_gru_cell_6_matmul_readvariableop_resource1while_gru_cell_6_matmul_readvariableop_resource_0"V
(while_gru_cell_6_readvariableop_resource*while_gru_cell_6_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_6/MatMul/ReadVariableOp&while/gru_cell_6/MatMul/ReadVariableOp2T
(while/gru_cell_6/MatMul_1/ReadVariableOp(while/gru_cell_6/MatMul_1/ReadVariableOp2B
while/gru_cell_6/ReadVariableOpwhile/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ð
¿
"__inference_internal_grad_fn_51844
result_grads_0
result_grads_10
,mul_sequential_2_gru_6_while_gru_cell_6_beta1
-mul_sequential_2_gru_6_while_gru_cell_6_add_2
identityª
mulMul,mul_sequential_2_gru_6_while_gru_cell_6_beta-mul_sequential_2_gru_6_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul,mul_sequential_2_gru_6_while_gru_cell_6_beta-mul_sequential_2_gru_6_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ø

"__inference_internal_grad_fn_52618
result_grads_0
result_grads_1
mul_gru_cell_7_beta
mul_gru_cell_7_add_2
identityx
mulMulmul_gru_cell_7_betamul_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_7_betamul_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

w
"__inference_internal_grad_fn_52924
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_49999
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_49999___redundant_placeholder03
/while_while_cond_49999___redundant_placeholder13
/while_while_cond_49999___redundant_placeholder23
/while_while_cond_49999___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ý

"__inference_internal_grad_fn_52636
result_grads_0
result_grads_1
mul_while_gru_cell_7_beta
mul_while_gru_cell_7_add_2
identity
mulMulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_45989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45989___redundant_placeholder03
/while_while_cond_45989___redundant_placeholder13
/while_while_cond_45989___redundant_placeholder23
/while_while_cond_45989___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ý

"__inference_internal_grad_fn_52096
result_grads_0
result_grads_1
mul_while_gru_cell_8_beta
mul_while_gru_cell_8_add_2
identity
mulMulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ô3
û
@__inference_gru_6_layer_call_and_return_conditional_losses_45865

inputs#
gru_cell_6_45789:	¬#
gru_cell_6_45791:	¬#
gru_cell_6_45793:	d¬
identity¢"gru_cell_6/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÀ
"gru_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_6_45789gru_cell_6_45791gru_cell_6_45793*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45788n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_6_45789gru_cell_6_45791gru_cell_6_45793*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_45801*
condR
while_cond_45800*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿds
NoOpNoOp#^gru_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"gru_cell_6/StatefulPartitionedCall"gru_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
!
Ù
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46492

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-46478*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdS
mul_3MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
ý

"__inference_internal_grad_fn_52132
result_grads_0
result_grads_1
mul_gru_7_gru_cell_7_beta
mul_gru_7_gru_cell_7_add_2
identity
mulMulmul_gru_7_gru_cell_7_betamul_gru_7_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_7_gru_cell_7_betamul_gru_7_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
§
¸
%__inference_gru_6_layer_call_fn_49184
inputs_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_45865|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ø

"__inference_internal_grad_fn_51934
result_grads_0
result_grads_1
mul_gru_cell_7_beta
mul_gru_cell_7_add_2
identityx
mulMulmul_gru_cell_7_betamul_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_7_betamul_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_46844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_6_readvariableop_resource_0:	¬D
1while_gru_cell_6_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_6_readvariableop_resource:	¬B
/while_gru_cell_6_matmul_readvariableop_resource:	¬D
1while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_6/MatMul/ReadVariableOp¢(while/gru_cell_6/MatMul_1/ReadVariableOp¢while/gru_cell_6/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_6/ReadVariableOpReadVariableOp*while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_6/unstackUnpack'while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_6/BiasAddBiasAdd!while/gru_cell_6/MatMul:product:0!while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_6/splitSplit)while/gru_cell_6/split/split_dim:output:0!while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_6/MatMul_1MatMulwhile_placeholder_20while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_6/BiasAdd_1BiasAdd#while/gru_cell_6/MatMul_1:product:0!while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_6/split_1SplitV#while/gru_cell_6/BiasAdd_1:output:0while/gru_cell_6/Const:output:0+while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_6/addAddV2while/gru_cell_6/split:output:0!while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_6/SigmoidSigmoidwhile/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_1AddV2while/gru_cell_6/split:output:1!while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_1Sigmoidwhile/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mulMulwhile/gru_cell_6/Sigmoid_1:y:0!while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_2AddV2while/gru_cell_6/split:output:2while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/mul_1Mulwhile/gru_cell_6/beta:output:0while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_2Sigmoidwhile/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_2Mulwhile/gru_cell_6/add_2:z:0while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/IdentityIdentitywhile/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_6/IdentityN	IdentityNwhile/gru_cell_6/mul_2:z:0while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-46894*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_3Mulwhile/gru_cell_6/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/subSubwhile/gru_cell_6/sub/x:output:0while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_4Mulwhile/gru_cell_6/sub:z:0#while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_3AddV2while/gru_cell_6/mul_3:z:0while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_6/MatMul/ReadVariableOp)^while/gru_cell_6/MatMul_1/ReadVariableOp ^while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_6_matmul_1_readvariableop_resource3while_gru_cell_6_matmul_1_readvariableop_resource_0"d
/while_gru_cell_6_matmul_readvariableop_resource1while_gru_cell_6_matmul_readvariableop_resource_0"V
(while_gru_cell_6_readvariableop_resource*while_gru_cell_6_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_6/MatMul/ReadVariableOp&while/gru_cell_6/MatMul/ReadVariableOp2T
(while/gru_cell_6/MatMul_1/ReadVariableOp(while/gru_cell_6/MatMul_1/ReadVariableOp2B
while/gru_cell_6/ReadVariableOpwhile/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ÆQ

@__inference_gru_6_layer_call_and_return_conditional_losses_47906

inputs5
"gru_cell_6_readvariableop_resource:	¬<
)gru_cell_6_matmul_readvariableop_resource:	¬>
+gru_cell_6_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_6/MatMul/ReadVariableOp¢"gru_cell_6/MatMul_1/ReadVariableOp¢gru_cell_6/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_6/ReadVariableOpReadVariableOp"gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_6/unstackUnpack!gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_6/MatMul/ReadVariableOpReadVariableOp)gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_6/MatMulMatMulstrided_slice_2:output:0(gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAddBiasAddgru_cell_6/MatMul:product:0gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_6/splitSplit#gru_cell_6/split/split_dim:output:0gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_6/MatMul_1MatMulzeros:output:0*gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAdd_1BiasAddgru_cell_6/MatMul_1:product:0gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_6/split_1SplitVgru_cell_6/BiasAdd_1:output:0gru_cell_6/Const:output:0%gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_6/addAddV2gru_cell_6/split:output:0gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_6/SigmoidSigmoidgru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_6/add_1AddV2gru_cell_6/split:output:1gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_1Sigmoidgru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_6/mulMulgru_cell_6/Sigmoid_1:y:0gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_6/add_2AddV2gru_cell_6/split:output:2gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_6/mul_1Mulgru_cell_6/beta:output:0gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_2Sigmoidgru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_6/mul_2Mulgru_cell_6/add_2:z:0gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/IdentityIdentitygru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_6/IdentityN	IdentityNgru_cell_6/mul_2:z:0gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47794*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_6/mul_3Mulgru_cell_6/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_6/subSubgru_cell_6/sub/x:output:0gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_6/mul_4Mulgru_cell_6/sub:z:0gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_6/add_3AddV2gru_cell_6/mul_3:z:0gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_6_readvariableop_resource)gru_cell_6_matmul_readvariableop_resource+gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_47810*
condR
while_cond_47809*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_6/MatMul/ReadVariableOp#^gru_cell_6/MatMul_1/ReadVariableOp^gru_cell_6/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_6/MatMul/ReadVariableOp gru_cell_6/MatMul/ReadVariableOp2H
"gru_cell_6/MatMul_1/ReadVariableOp"gru_cell_6/MatMul_1/ReadVariableOp26
gru_cell_6/ReadVariableOpgru_cell_6/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ø

"__inference_internal_grad_fn_52042
result_grads_0
result_grads_1
mul_gru_cell_7_beta
mul_gru_cell_7_add_2
identityx
mulMulmul_gru_cell_7_betamul_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_7_betamul_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ô3
û
@__inference_gru_6_layer_call_and_return_conditional_losses_46054

inputs#
gru_cell_6_45978:	¬#
gru_cell_6_45980:	¬#
gru_cell_6_45982:	d¬
identity¢"gru_cell_6/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÀ
"gru_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_6_45978gru_cell_6_45980gru_cell_6_45982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45938n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_6_45978gru_cell_6_45980gru_cell_6_45982*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_45990*
condR
while_cond_45989*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿds
NoOpNoOp#^gru_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"gru_cell_6/StatefulPartitionedCall"gru_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_49454
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_49454___redundant_placeholder03
/while_while_cond_49454___redundant_placeholder13
/while_while_cond_49454___redundant_placeholder23
/while_while_cond_49454___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:

w
"__inference_internal_grad_fn_52348
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
æ[
¸
#sequential_2_gru_8_while_body_45609B
>sequential_2_gru_8_while_sequential_2_gru_8_while_loop_counterH
Dsequential_2_gru_8_while_sequential_2_gru_8_while_maximum_iterations(
$sequential_2_gru_8_while_placeholder*
&sequential_2_gru_8_while_placeholder_1*
&sequential_2_gru_8_while_placeholder_2A
=sequential_2_gru_8_while_sequential_2_gru_8_strided_slice_1_0}
ysequential_2_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_8_tensorarrayunstack_tensorlistfromtensor_0P
=sequential_2_gru_8_while_gru_cell_8_readvariableop_resource_0:	¬W
Dsequential_2_gru_8_while_gru_cell_8_matmul_readvariableop_resource_0:	d¬Y
Fsequential_2_gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬%
!sequential_2_gru_8_while_identity'
#sequential_2_gru_8_while_identity_1'
#sequential_2_gru_8_while_identity_2'
#sequential_2_gru_8_while_identity_3'
#sequential_2_gru_8_while_identity_4?
;sequential_2_gru_8_while_sequential_2_gru_8_strided_slice_1{
wsequential_2_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_8_tensorarrayunstack_tensorlistfromtensorN
;sequential_2_gru_8_while_gru_cell_8_readvariableop_resource:	¬U
Bsequential_2_gru_8_while_gru_cell_8_matmul_readvariableop_resource:	d¬W
Dsequential_2_gru_8_while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢9sequential_2/gru_8/while/gru_cell_8/MatMul/ReadVariableOp¢;sequential_2/gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp¢2sequential_2/gru_8/while/gru_cell_8/ReadVariableOp
Jsequential_2/gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
<sequential_2/gru_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_2_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_8_tensorarrayunstack_tensorlistfromtensor_0$sequential_2_gru_8_while_placeholderSsequential_2/gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0±
2sequential_2/gru_8/while/gru_cell_8/ReadVariableOpReadVariableOp=sequential_2_gru_8_while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0©
+sequential_2/gru_8/while/gru_cell_8/unstackUnpack:sequential_2/gru_8/while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¿
9sequential_2/gru_8/while/gru_cell_8/MatMul/ReadVariableOpReadVariableOpDsequential_2_gru_8_while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0ï
*sequential_2/gru_8/while/gru_cell_8/MatMulMatMulCsequential_2/gru_8/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_2/gru_8/while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Õ
+sequential_2/gru_8/while/gru_cell_8/BiasAddBiasAdd4sequential_2/gru_8/while/gru_cell_8/MatMul:product:04sequential_2/gru_8/while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
3sequential_2/gru_8/while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
)sequential_2/gru_8/while/gru_cell_8/splitSplit<sequential_2/gru_8/while/gru_cell_8/split/split_dim:output:04sequential_2/gru_8/while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÃ
;sequential_2/gru_8/while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOpFsequential_2_gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ö
,sequential_2/gru_8/while/gru_cell_8/MatMul_1MatMul&sequential_2_gru_8_while_placeholder_2Csequential_2/gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ù
-sequential_2/gru_8/while/gru_cell_8/BiasAdd_1BiasAdd6sequential_2/gru_8/while/gru_cell_8/MatMul_1:product:04sequential_2/gru_8/while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
)sequential_2/gru_8/while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ
5sequential_2/gru_8/while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
+sequential_2/gru_8/while/gru_cell_8/split_1SplitV6sequential_2/gru_8/while/gru_cell_8/BiasAdd_1:output:02sequential_2/gru_8/while/gru_cell_8/Const:output:0>sequential_2/gru_8/while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÌ
'sequential_2/gru_8/while/gru_cell_8/addAddV22sequential_2/gru_8/while/gru_cell_8/split:output:04sequential_2/gru_8/while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+sequential_2/gru_8/while/gru_cell_8/SigmoidSigmoid+sequential_2/gru_8/while/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÎ
)sequential_2/gru_8/while/gru_cell_8/add_1AddV22sequential_2/gru_8/while/gru_cell_8/split:output:14sequential_2/gru_8/while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_8/while/gru_cell_8/Sigmoid_1Sigmoid-sequential_2/gru_8/while/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÉ
'sequential_2/gru_8/while/gru_cell_8/mulMul1sequential_2/gru_8/while/gru_cell_8/Sigmoid_1:y:04sequential_2/gru_8/while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ
)sequential_2/gru_8/while/gru_cell_8/add_2AddV22sequential_2/gru_8/while/gru_cell_8/split:output:2+sequential_2/gru_8/while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(sequential_2/gru_8/while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
)sequential_2/gru_8/while/gru_cell_8/mul_1Mul1sequential_2/gru_8/while/gru_cell_8/beta:output:0-sequential_2/gru_8/while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_8/while/gru_cell_8/Sigmoid_2Sigmoid-sequential_2/gru_8/while/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
)sequential_2/gru_8/while/gru_cell_8/mul_2Mul-sequential_2/gru_8/while/gru_cell_8/add_2:z:01sequential_2/gru_8/while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_2/gru_8/while/gru_cell_8/IdentityIdentity-sequential_2/gru_8/while/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_8/while/gru_cell_8/IdentityN	IdentityN-sequential_2/gru_8/while/gru_cell_8/mul_2:z:0-sequential_2/gru_8/while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45659*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd»
)sequential_2/gru_8/while/gru_cell_8/mul_3Mul/sequential_2/gru_8/while/gru_cell_8/Sigmoid:y:0&sequential_2_gru_8_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
)sequential_2/gru_8/while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Å
'sequential_2/gru_8/while/gru_cell_8/subSub2sequential_2/gru_8/while/gru_cell_8/sub/x:output:0/sequential_2/gru_8/while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
)sequential_2/gru_8/while/gru_cell_8/mul_4Mul+sequential_2/gru_8/while/gru_cell_8/sub:z:06sequential_2/gru_8/while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ
)sequential_2/gru_8/while/gru_cell_8/add_3AddV2-sequential_2/gru_8/while/gru_cell_8/mul_3:z:0-sequential_2/gru_8/while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
=sequential_2/gru_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_2_gru_8_while_placeholder_1$sequential_2_gru_8_while_placeholder-sequential_2/gru_8/while/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_2/gru_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_2/gru_8/while/addAddV2$sequential_2_gru_8_while_placeholder'sequential_2/gru_8/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_2/gru_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_2/gru_8/while/add_1AddV2>sequential_2_gru_8_while_sequential_2_gru_8_while_loop_counter)sequential_2/gru_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_2/gru_8/while/IdentityIdentity"sequential_2/gru_8/while/add_1:z:0^sequential_2/gru_8/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_2/gru_8/while/Identity_1IdentityDsequential_2_gru_8_while_sequential_2_gru_8_while_maximum_iterations^sequential_2/gru_8/while/NoOp*
T0*
_output_shapes
: 
#sequential_2/gru_8/while/Identity_2Identity sequential_2/gru_8/while/add:z:0^sequential_2/gru_8/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_2/gru_8/while/Identity_3IdentityMsequential_2/gru_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/gru_8/while/NoOp*
T0*
_output_shapes
: :éèÒ°
#sequential_2/gru_8/while/Identity_4Identity-sequential_2/gru_8/while/gru_cell_8/add_3:z:0^sequential_2/gru_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_2/gru_8/while/NoOpNoOp:^sequential_2/gru_8/while/gru_cell_8/MatMul/ReadVariableOp<^sequential_2/gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp3^sequential_2/gru_8/while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Dsequential_2_gru_8_while_gru_cell_8_matmul_1_readvariableop_resourceFsequential_2_gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0"
Bsequential_2_gru_8_while_gru_cell_8_matmul_readvariableop_resourceDsequential_2_gru_8_while_gru_cell_8_matmul_readvariableop_resource_0"|
;sequential_2_gru_8_while_gru_cell_8_readvariableop_resource=sequential_2_gru_8_while_gru_cell_8_readvariableop_resource_0"O
!sequential_2_gru_8_while_identity*sequential_2/gru_8/while/Identity:output:0"S
#sequential_2_gru_8_while_identity_1,sequential_2/gru_8/while/Identity_1:output:0"S
#sequential_2_gru_8_while_identity_2,sequential_2/gru_8/while/Identity_2:output:0"S
#sequential_2_gru_8_while_identity_3,sequential_2/gru_8/while/Identity_3:output:0"S
#sequential_2_gru_8_while_identity_4,sequential_2/gru_8/while/Identity_4:output:0"|
;sequential_2_gru_8_while_sequential_2_gru_8_strided_slice_1=sequential_2_gru_8_while_sequential_2_gru_8_strided_slice_1_0"ô
wsequential_2_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_8_tensorarrayunstack_tensorlistfromtensorysequential_2_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2v
9sequential_2/gru_8/while/gru_cell_8/MatMul/ReadVariableOp9sequential_2/gru_8/while/gru_cell_8/MatMul/ReadVariableOp2z
;sequential_2/gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp;sequential_2/gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp2h
2sequential_2/gru_8/while/gru_cell_8/ReadVariableOp2sequential_2/gru_8/while/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
§
¸
%__inference_gru_7_layer_call_fn_49896
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_46217|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
å
§
while_body_46505
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_8_46527_0:	¬+
while_gru_cell_8_46529_0:	d¬+
while_gru_cell_8_46531_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_8_46527:	¬)
while_gru_cell_8_46529:	d¬)
while_gru_cell_8_46531:	d¬¢(while/gru_cell_8/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0û
(while/gru_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_8_46527_0while_gru_cell_8_46529_0while_gru_cell_8_46531_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46492Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_8_46527while_gru_cell_8_46527_0"2
while_gru_cell_8_46529while_gru_cell_8_46529_0"2
while_gru_cell_8_46531while_gru_cell_8_46531_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/gru_cell_8/StatefulPartitionedCall(while/gru_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
J
²	
gru_7_while_body_48879(
$gru_7_while_gru_7_while_loop_counter.
*gru_7_while_gru_7_while_maximum_iterations
gru_7_while_placeholder
gru_7_while_placeholder_1
gru_7_while_placeholder_2'
#gru_7_while_gru_7_strided_slice_1_0c
_gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_0C
0gru_7_while_gru_cell_7_readvariableop_resource_0:	¬J
7gru_7_while_gru_cell_7_matmul_readvariableop_resource_0:	d¬L
9gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
gru_7_while_identity
gru_7_while_identity_1
gru_7_while_identity_2
gru_7_while_identity_3
gru_7_while_identity_4%
!gru_7_while_gru_7_strided_slice_1a
]gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensorA
.gru_7_while_gru_cell_7_readvariableop_resource:	¬H
5gru_7_while_gru_cell_7_matmul_readvariableop_resource:	d¬J
7gru_7_while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢,gru_7/while/gru_cell_7/MatMul/ReadVariableOp¢.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp¢%gru_7/while/gru_cell_7/ReadVariableOp
=gru_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_0gru_7_while_placeholderFgru_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
%gru_7/while/gru_cell_7/ReadVariableOpReadVariableOp0gru_7_while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_7/while/gru_cell_7/unstackUnpack-gru_7/while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_7/while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp7gru_7_while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0È
gru_7/while/gru_cell_7/MatMulMatMul6gru_7/while/TensorArrayV2Read/TensorListGetItem:item:04gru_7/while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_7/while/gru_cell_7/BiasAddBiasAdd'gru_7/while/gru_cell_7/MatMul:product:0'gru_7/while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_7/while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_7/while/gru_cell_7/splitSplit/gru_7/while/gru_cell_7/split/split_dim:output:0'gru_7/while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp9gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_7/while/gru_cell_7/MatMul_1MatMulgru_7_while_placeholder_26gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_7/while/gru_cell_7/BiasAdd_1BiasAdd)gru_7/while/gru_cell_7/MatMul_1:product:0'gru_7/while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_7/while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_7/while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_7/while/gru_cell_7/split_1SplitV)gru_7/while/gru_cell_7/BiasAdd_1:output:0%gru_7/while/gru_cell_7/Const:output:01gru_7/while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_7/while/gru_cell_7/addAddV2%gru_7/while/gru_cell_7/split:output:0'gru_7/while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_7/while/gru_cell_7/SigmoidSigmoidgru_7/while/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_7/while/gru_cell_7/add_1AddV2%gru_7/while/gru_cell_7/split:output:1'gru_7/while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_7/while/gru_cell_7/Sigmoid_1Sigmoid gru_7/while/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_7/while/gru_cell_7/mulMul$gru_7/while/gru_cell_7/Sigmoid_1:y:0'gru_7/while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/add_2AddV2%gru_7/while/gru_cell_7/split:output:2gru_7/while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_7/while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/while/gru_cell_7/mul_1Mul$gru_7/while/gru_cell_7/beta:output:0 gru_7/while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_7/while/gru_cell_7/Sigmoid_2Sigmoid gru_7/while/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/mul_2Mul gru_7/while/gru_cell_7/add_2:z:0$gru_7/while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/IdentityIdentity gru_7/while/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdç
 gru_7/while/gru_cell_7/IdentityN	IdentityN gru_7/while/gru_cell_7/mul_2:z:0 gru_7/while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48929*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/mul_3Mul"gru_7/while/gru_cell_7/Sigmoid:y:0gru_7_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_7/while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/while/gru_cell_7/subSub%gru_7/while/gru_cell_7/sub/x:output:0"gru_7/while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_7/while/gru_cell_7/mul_4Mulgru_7/while/gru_cell_7/sub:z:0)gru_7/while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/add_3AddV2 gru_7/while/gru_cell_7/mul_3:z:0 gru_7/while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_7_while_placeholder_1gru_7_while_placeholder gru_7/while/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_7/while/addAddV2gru_7_while_placeholdergru_7/while/add/y:output:0*
T0*
_output_shapes
: U
gru_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_7/while/add_1AddV2$gru_7_while_gru_7_while_loop_countergru_7/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_7/while/IdentityIdentitygru_7/while/add_1:z:0^gru_7/while/NoOp*
T0*
_output_shapes
: 
gru_7/while/Identity_1Identity*gru_7_while_gru_7_while_maximum_iterations^gru_7/while/NoOp*
T0*
_output_shapes
: k
gru_7/while/Identity_2Identitygru_7/while/add:z:0^gru_7/while/NoOp*
T0*
_output_shapes
: «
gru_7/while/Identity_3Identity@gru_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_7/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_7/while/Identity_4Identity gru_7/while/gru_cell_7/add_3:z:0^gru_7/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_7/while/NoOpNoOp-^gru_7/while/gru_cell_7/MatMul/ReadVariableOp/^gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp&^gru_7/while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_7_while_gru_7_strided_slice_1#gru_7_while_gru_7_strided_slice_1_0"t
7gru_7_while_gru_cell_7_matmul_1_readvariableop_resource9gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0"p
5gru_7_while_gru_cell_7_matmul_readvariableop_resource7gru_7_while_gru_cell_7_matmul_readvariableop_resource_0"b
.gru_7_while_gru_cell_7_readvariableop_resource0gru_7_while_gru_cell_7_readvariableop_resource_0"5
gru_7_while_identitygru_7/while/Identity:output:0"9
gru_7_while_identity_1gru_7/while/Identity_1:output:0"9
gru_7_while_identity_2gru_7/while/Identity_2:output:0"9
gru_7_while_identity_3gru_7/while/Identity_3:output:0"9
gru_7_while_identity_4gru_7/while/Identity_4:output:0"À
]gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_7/while/gru_cell_7/MatMul/ReadVariableOp,gru_7/while/gru_cell_7/MatMul/ReadVariableOp2`
.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp2N
%gru_7/while/gru_cell_7/ReadVariableOp%gru_7/while/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Ä

£
#__inference_signature_wrapper_49173
gru_6_input
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
	unknown_2:	¬
	unknown_3:	d¬
	unknown_4:	d¬
	unknown_5:	¬
	unknown_6:	d¬
	unknown_7:	d¬
	unknown_8:d
	unknown_9:
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallgru_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_45711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_6_input
×

#sequential_2_gru_6_while_cond_45282B
>sequential_2_gru_6_while_sequential_2_gru_6_while_loop_counterH
Dsequential_2_gru_6_while_sequential_2_gru_6_while_maximum_iterations(
$sequential_2_gru_6_while_placeholder*
&sequential_2_gru_6_while_placeholder_1*
&sequential_2_gru_6_while_placeholder_2D
@sequential_2_gru_6_while_less_sequential_2_gru_6_strided_slice_1Y
Usequential_2_gru_6_while_sequential_2_gru_6_while_cond_45282___redundant_placeholder0Y
Usequential_2_gru_6_while_sequential_2_gru_6_while_cond_45282___redundant_placeholder1Y
Usequential_2_gru_6_while_sequential_2_gru_6_while_cond_45282___redundant_placeholder2Y
Usequential_2_gru_6_while_sequential_2_gru_6_while_cond_45282___redundant_placeholder3%
!sequential_2_gru_6_while_identity
®
sequential_2/gru_6/while/LessLess$sequential_2_gru_6_while_placeholder@sequential_2_gru_6_while_less_sequential_2_gru_6_strided_slice_1*
T0*
_output_shapes
: q
!sequential_2/gru_6/while/IdentityIdentity!sequential_2/gru_6/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_2_gru_6_while_identity*sequential_2/gru_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ì
³
"__inference_internal_grad_fn_51790
result_grads_0
result_grads_1*
&mul_sequential_2_gru_6_gru_cell_6_beta+
'mul_sequential_2_gru_6_gru_cell_6_add_2
identity
mulMul&mul_sequential_2_gru_6_gru_cell_6_beta'mul_sequential_2_gru_6_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul&mul_sequential_2_gru_6_gru_cell_6_beta'mul_sequential_2_gru_6_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ÆQ

@__inference_gru_7_layer_call_and_return_conditional_losses_47717

inputs5
"gru_cell_7_readvariableop_resource:	¬<
)gru_cell_7_matmul_readvariableop_resource:	d¬>
+gru_cell_7_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_7/MatMul/ReadVariableOp¢"gru_cell_7/MatMul_1/ReadVariableOp¢gru_cell_7/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_7/ReadVariableOpReadVariableOp"gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_7/unstackUnpack!gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_7/MatMul/ReadVariableOpReadVariableOp)gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMulMatMulstrided_slice_2:output:0(gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAddBiasAddgru_cell_7/MatMul:product:0gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_7/splitSplit#gru_cell_7/split/split_dim:output:0gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMul_1MatMulzeros:output:0*gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAdd_1BiasAddgru_cell_7/MatMul_1:product:0gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_7/split_1SplitVgru_cell_7/BiasAdd_1:output:0gru_cell_7/Const:output:0%gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_7/addAddV2gru_cell_7/split:output:0gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_7/SigmoidSigmoidgru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_7/add_1AddV2gru_cell_7/split:output:1gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_1Sigmoidgru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_7/mulMulgru_cell_7/Sigmoid_1:y:0gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_7/add_2AddV2gru_cell_7/split:output:2gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_7/mul_1Mulgru_cell_7/beta:output:0gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_2Sigmoidgru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_7/mul_2Mulgru_cell_7/add_2:z:0gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/IdentityIdentitygru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_7/IdentityN	IdentityNgru_cell_7/mul_2:z:0gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47605*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_7/mul_3Mulgru_cell_7/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_7/subSubgru_cell_7/sub/x:output:0gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_7/mul_4Mulgru_cell_7/sub:z:0gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_7/add_3AddV2gru_cell_7/mul_3:z:0gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_7_readvariableop_resource)gru_cell_7_matmul_readvariableop_resource+gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_47621*
condR
while_cond_47620*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_7/MatMul/ReadVariableOp#^gru_cell_7/MatMul_1/ReadVariableOp^gru_cell_7/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_7/MatMul/ReadVariableOp gru_cell_7/MatMul/ReadVariableOp2H
"gru_cell_7/MatMul_1/ReadVariableOp"gru_cell_7/MatMul_1/ReadVariableOp26
gru_cell_7/ReadVariableOpgru_cell_7/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ÆQ

@__inference_gru_6_layer_call_and_return_conditional_losses_49885

inputs5
"gru_cell_6_readvariableop_resource:	¬<
)gru_cell_6_matmul_readvariableop_resource:	¬>
+gru_cell_6_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_6/MatMul/ReadVariableOp¢"gru_cell_6/MatMul_1/ReadVariableOp¢gru_cell_6/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_6/ReadVariableOpReadVariableOp"gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_6/unstackUnpack!gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_6/MatMul/ReadVariableOpReadVariableOp)gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_6/MatMulMatMulstrided_slice_2:output:0(gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAddBiasAddgru_cell_6/MatMul:product:0gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_6/splitSplit#gru_cell_6/split/split_dim:output:0gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_6/MatMul_1MatMulzeros:output:0*gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAdd_1BiasAddgru_cell_6/MatMul_1:product:0gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_6/split_1SplitVgru_cell_6/BiasAdd_1:output:0gru_cell_6/Const:output:0%gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_6/addAddV2gru_cell_6/split:output:0gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_6/SigmoidSigmoidgru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_6/add_1AddV2gru_cell_6/split:output:1gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_1Sigmoidgru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_6/mulMulgru_cell_6/Sigmoid_1:y:0gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_6/add_2AddV2gru_cell_6/split:output:2gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_6/mul_1Mulgru_cell_6/beta:output:0gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_2Sigmoidgru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_6/mul_2Mulgru_cell_6/add_2:z:0gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/IdentityIdentitygru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_6/IdentityN	IdentityNgru_cell_6/mul_2:z:0gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49773*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_6/mul_3Mulgru_cell_6/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_6/subSubgru_cell_6/sub/x:output:0gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_6/mul_4Mulgru_cell_6/sub:z:0gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_6/add_3AddV2gru_cell_6/mul_3:z:0gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_6_readvariableop_resource)gru_cell_6_matmul_readvariableop_resource+gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_49789*
condR
while_cond_49788*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_6/MatMul/ReadVariableOp#^gru_cell_6/MatMul_1/ReadVariableOp^gru_cell_6/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_6/MatMul/ReadVariableOp gru_cell_6/MatMul/ReadVariableOp2H
"gru_cell_6/MatMul_1/ReadVariableOp"gru_cell_6/MatMul_1/ReadVariableOp26
gru_cell_6/ReadVariableOpgru_cell_6/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
J
²	
gru_8_while_body_49042(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2'
#gru_8_while_gru_8_strided_slice_1_0c
_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0C
0gru_8_while_gru_cell_8_readvariableop_resource_0:	¬J
7gru_8_while_gru_cell_8_matmul_readvariableop_resource_0:	d¬L
9gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
gru_8_while_identity
gru_8_while_identity_1
gru_8_while_identity_2
gru_8_while_identity_3
gru_8_while_identity_4%
!gru_8_while_gru_8_strided_slice_1a
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensorA
.gru_8_while_gru_cell_8_readvariableop_resource:	¬H
5gru_8_while_gru_cell_8_matmul_readvariableop_resource:	d¬J
7gru_8_while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢,gru_8/while/gru_cell_8/MatMul/ReadVariableOp¢.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp¢%gru_8/while/gru_cell_8/ReadVariableOp
=gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0gru_8_while_placeholderFgru_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
%gru_8/while/gru_cell_8/ReadVariableOpReadVariableOp0gru_8_while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_8/while/gru_cell_8/unstackUnpack-gru_8/while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_8/while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp7gru_8_while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0È
gru_8/while/gru_cell_8/MatMulMatMul6gru_8/while/TensorArrayV2Read/TensorListGetItem:item:04gru_8/while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_8/while/gru_cell_8/BiasAddBiasAdd'gru_8/while/gru_cell_8/MatMul:product:0'gru_8/while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_8/while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_8/while/gru_cell_8/splitSplit/gru_8/while/gru_cell_8/split/split_dim:output:0'gru_8/while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp9gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_8/while/gru_cell_8/MatMul_1MatMulgru_8_while_placeholder_26gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_8/while/gru_cell_8/BiasAdd_1BiasAdd)gru_8/while/gru_cell_8/MatMul_1:product:0'gru_8/while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_8/while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_8/while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_8/while/gru_cell_8/split_1SplitV)gru_8/while/gru_cell_8/BiasAdd_1:output:0%gru_8/while/gru_cell_8/Const:output:01gru_8/while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_8/while/gru_cell_8/addAddV2%gru_8/while/gru_cell_8/split:output:0'gru_8/while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_8/while/gru_cell_8/SigmoidSigmoidgru_8/while/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_8/while/gru_cell_8/add_1AddV2%gru_8/while/gru_cell_8/split:output:1'gru_8/while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_8/while/gru_cell_8/Sigmoid_1Sigmoid gru_8/while/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_8/while/gru_cell_8/mulMul$gru_8/while/gru_cell_8/Sigmoid_1:y:0'gru_8/while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/add_2AddV2%gru_8/while/gru_cell_8/split:output:2gru_8/while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_8/while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/while/gru_cell_8/mul_1Mul$gru_8/while/gru_cell_8/beta:output:0 gru_8/while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_8/while/gru_cell_8/Sigmoid_2Sigmoid gru_8/while/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/mul_2Mul gru_8/while/gru_cell_8/add_2:z:0$gru_8/while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/IdentityIdentity gru_8/while/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdç
 gru_8/while/gru_cell_8/IdentityN	IdentityN gru_8/while/gru_cell_8/mul_2:z:0 gru_8/while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49092*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/mul_3Mul"gru_8/while/gru_cell_8/Sigmoid:y:0gru_8_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_8/while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/while/gru_cell_8/subSub%gru_8/while/gru_cell_8/sub/x:output:0"gru_8/while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_8/while/gru_cell_8/mul_4Mulgru_8/while/gru_cell_8/sub:z:0)gru_8/while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/add_3AddV2 gru_8/while/gru_cell_8/mul_3:z:0 gru_8/while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_8_while_placeholder_1gru_8_while_placeholder gru_8/while/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_8/while/addAddV2gru_8_while_placeholdergru_8/while/add/y:output:0*
T0*
_output_shapes
: U
gru_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_8/while/add_1AddV2$gru_8_while_gru_8_while_loop_countergru_8/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_8/while/IdentityIdentitygru_8/while/add_1:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: 
gru_8/while/Identity_1Identity*gru_8_while_gru_8_while_maximum_iterations^gru_8/while/NoOp*
T0*
_output_shapes
: k
gru_8/while/Identity_2Identitygru_8/while/add:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: «
gru_8/while/Identity_3Identity@gru_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_8/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_8/while/Identity_4Identity gru_8/while/gru_cell_8/add_3:z:0^gru_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_8/while/NoOpNoOp-^gru_8/while/gru_cell_8/MatMul/ReadVariableOp/^gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp&^gru_8/while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_8_while_gru_8_strided_slice_1#gru_8_while_gru_8_strided_slice_1_0"t
7gru_8_while_gru_cell_8_matmul_1_readvariableop_resource9gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0"p
5gru_8_while_gru_cell_8_matmul_readvariableop_resource7gru_8_while_gru_cell_8_matmul_readvariableop_resource_0"b
.gru_8_while_gru_cell_8_readvariableop_resource0gru_8_while_gru_cell_8_readvariableop_resource_0"5
gru_8_while_identitygru_8/while/Identity:output:0"9
gru_8_while_identity_1gru_8/while/Identity_1:output:0"9
gru_8_while_identity_2gru_8/while/Identity_2:output:0"9
gru_8_while_identity_3gru_8/while/Identity_3:output:0"9
gru_8_while_identity_4gru_8/while/Identity_4:output:0"À
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_8/while/gru_cell_8/MatMul/ReadVariableOp,gru_8/while/gru_cell_8/MatMul/ReadVariableOp2`
.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp2N
%gru_8/while/gru_cell_8/ReadVariableOp%gru_8/while/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 

¸
%__inference_gru_8_layer_call_fn_50608
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_46569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
Õ
¥
while_cond_50500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_50500___redundant_placeholder03
/while_while_cond_50500___redundant_placeholder13
/while_while_cond_50500___redundant_placeholder23
/while_while_cond_50500___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ø

"__inference_internal_grad_fn_52798
result_grads_0
result_grads_1
mul_gru_cell_8_beta
mul_gru_cell_8_add_2
identityx
mulMulmul_gru_cell_8_betamul_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_8_betamul_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_47620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47620___redundant_placeholder03
/while_while_cond_47620___redundant_placeholder13
/while_while_cond_47620___redundant_placeholder23
/while_while_cond_47620___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
B
þ
while_body_47432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_8_readvariableop_resource_0:	¬D
1while_gru_cell_8_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_8_readvariableop_resource:	¬B
/while_gru_cell_8_matmul_readvariableop_resource:	d¬D
1while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_8/MatMul/ReadVariableOp¢(while/gru_cell_8/MatMul_1/ReadVariableOp¢while/gru_cell_8/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_8/ReadVariableOpReadVariableOp*while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_8/unstackUnpack'while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_8/BiasAddBiasAdd!while/gru_cell_8/MatMul:product:0!while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_8/splitSplit)while/gru_cell_8/split/split_dim:output:0!while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_8/MatMul_1MatMulwhile_placeholder_20while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_8/BiasAdd_1BiasAdd#while/gru_cell_8/MatMul_1:product:0!while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_8/split_1SplitV#while/gru_cell_8/BiasAdd_1:output:0while/gru_cell_8/Const:output:0+while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_8/addAddV2while/gru_cell_8/split:output:0!while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_8/SigmoidSigmoidwhile/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_1AddV2while/gru_cell_8/split:output:1!while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_1Sigmoidwhile/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mulMulwhile/gru_cell_8/Sigmoid_1:y:0!while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_2AddV2while/gru_cell_8/split:output:2while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/mul_1Mulwhile/gru_cell_8/beta:output:0while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_2Sigmoidwhile/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_2Mulwhile/gru_cell_8/add_2:z:0while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/IdentityIdentitywhile/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_8/IdentityN	IdentityNwhile/gru_cell_8/mul_2:z:0while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47482*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_3Mulwhile/gru_cell_8/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/subSubwhile/gru_cell_8/sub/x:output:0while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_4Mulwhile/gru_cell_8/sub:z:0#while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_3AddV2while/gru_cell_8/mul_3:z:0while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_8/MatMul/ReadVariableOp)^while/gru_cell_8/MatMul_1/ReadVariableOp ^while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_8_matmul_1_readvariableop_resource3while_gru_cell_8_matmul_1_readvariableop_resource_0"d
/while_gru_cell_8_matmul_readvariableop_resource1while_gru_cell_8_matmul_readvariableop_resource_0"V
(while_gru_cell_8_readvariableop_resource*while_gru_cell_8_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_8/MatMul/ReadVariableOp&while/gru_cell_8/MatMul/ReadVariableOp2T
(while/gru_cell_8/MatMul_1/ReadVariableOp(while/gru_cell_8/MatMul_1/ReadVariableOp2B
while/gru_cell_8/ReadVariableOpwhile/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 

w
"__inference_internal_grad_fn_52708
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_45800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45800___redundant_placeholder03
/while_while_cond_45800___redundant_placeholder13
/while_while_cond_45800___redundant_placeholder23
/while_while_cond_45800___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ý

"__inference_internal_grad_fn_52816
result_grads_0
result_grads_1
mul_while_gru_cell_8_beta
mul_while_gru_cell_8_add_2
identity
mulMulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
R

@__inference_gru_8_layer_call_and_return_conditional_losses_50808
inputs_05
"gru_cell_8_readvariableop_resource:	¬<
)gru_cell_8_matmul_readvariableop_resource:	d¬>
+gru_cell_8_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_8/MatMul/ReadVariableOp¢"gru_cell_8/MatMul_1/ReadVariableOp¢gru_cell_8/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_8/ReadVariableOpReadVariableOp"gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_8/unstackUnpack!gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_8/MatMul/ReadVariableOpReadVariableOp)gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMulMatMulstrided_slice_2:output:0(gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAddBiasAddgru_cell_8/MatMul:product:0gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_8/splitSplit#gru_cell_8/split/split_dim:output:0gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMul_1MatMulzeros:output:0*gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAdd_1BiasAddgru_cell_8/MatMul_1:product:0gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_8/split_1SplitVgru_cell_8/BiasAdd_1:output:0gru_cell_8/Const:output:0%gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_8/addAddV2gru_cell_8/split:output:0gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_8/SigmoidSigmoidgru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_8/add_1AddV2gru_cell_8/split:output:1gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_1Sigmoidgru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_8/mulMulgru_cell_8/Sigmoid_1:y:0gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_8/add_2AddV2gru_cell_8/split:output:2gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_8/mul_1Mulgru_cell_8/beta:output:0gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_2Sigmoidgru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_8/mul_2Mulgru_cell_8/add_2:z:0gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/IdentityIdentitygru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_8/IdentityN	IdentityNgru_cell_8/mul_2:z:0gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50696*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_8/mul_3Mulgru_cell_8/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_8/subSubgru_cell_8/sub/x:output:0gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_8/mul_4Mulgru_cell_8/sub:z:0gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_8/add_3AddV2gru_cell_8/mul_3:z:0gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_8_readvariableop_resource)gru_cell_8_matmul_readvariableop_resource+gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_50712*
condR
while_cond_50711*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_8/MatMul/ReadVariableOp#^gru_cell_8/MatMul_1/ReadVariableOp^gru_cell_8/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_8/MatMul/ReadVariableOp gru_cell_8/MatMul/ReadVariableOp2H
"gru_cell_8/MatMul_1/ReadVariableOp"gru_cell_8/MatMul_1/ReadVariableOp26
gru_cell_8/ReadVariableOpgru_cell_8/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
ÆQ

@__inference_gru_6_layer_call_and_return_conditional_losses_49718

inputs5
"gru_cell_6_readvariableop_resource:	¬<
)gru_cell_6_matmul_readvariableop_resource:	¬>
+gru_cell_6_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_6/MatMul/ReadVariableOp¢"gru_cell_6/MatMul_1/ReadVariableOp¢gru_cell_6/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_6/ReadVariableOpReadVariableOp"gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_6/unstackUnpack!gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_6/MatMul/ReadVariableOpReadVariableOp)gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_6/MatMulMatMulstrided_slice_2:output:0(gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAddBiasAddgru_cell_6/MatMul:product:0gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_6/splitSplit#gru_cell_6/split/split_dim:output:0gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_6/MatMul_1MatMulzeros:output:0*gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAdd_1BiasAddgru_cell_6/MatMul_1:product:0gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_6/split_1SplitVgru_cell_6/BiasAdd_1:output:0gru_cell_6/Const:output:0%gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_6/addAddV2gru_cell_6/split:output:0gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_6/SigmoidSigmoidgru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_6/add_1AddV2gru_cell_6/split:output:1gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_1Sigmoidgru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_6/mulMulgru_cell_6/Sigmoid_1:y:0gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_6/add_2AddV2gru_cell_6/split:output:2gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_6/mul_1Mulgru_cell_6/beta:output:0gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_2Sigmoidgru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_6/mul_2Mulgru_cell_6/add_2:z:0gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/IdentityIdentitygru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_6/IdentityN	IdentityNgru_cell_6/mul_2:z:0gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49606*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_6/mul_3Mulgru_cell_6/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_6/subSubgru_cell_6/sub/x:output:0gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_6/mul_4Mulgru_cell_6/sub:z:0gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_6/add_3AddV2gru_cell_6/mul_3:z:0gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_6_readvariableop_resource)gru_cell_6_matmul_readvariableop_resource+gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_49622*
condR
while_cond_49621*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_6/MatMul/ReadVariableOp#^gru_cell_6/MatMul_1/ReadVariableOp^gru_cell_6/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_6/MatMul/ReadVariableOp gru_cell_6/MatMul/ReadVariableOp2H
"gru_cell_6/MatMul_1/ReadVariableOp"gru_cell_6/MatMul_1/ReadVariableOp26
gru_cell_6/ReadVariableOpgru_cell_6/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Õ
¥
while_cond_51045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_51045___redundant_placeholder03
/while_while_cond_51045___redundant_placeholder13
/while_while_cond_51045___redundant_placeholder23
/while_while_cond_51045___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ø

"__inference_internal_grad_fn_52006
result_grads_0
result_grads_1
mul_gru_cell_6_beta
mul_gru_cell_6_add_2
identityx
mulMulmul_gru_cell_6_betamul_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_6_betamul_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_50878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_50878___redundant_placeholder03
/while_while_cond_50878___redundant_placeholder13
/while_while_cond_50878___redundant_placeholder23
/while_while_cond_50878___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
°

Ù
*__inference_gru_cell_6_layer_call_fn_51342

inputs
states_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45788o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
ý

"__inference_internal_grad_fn_52150
result_grads_0
result_grads_1
mul_gru_8_gru_cell_8_beta
mul_gru_8_gru_cell_8_add_2
identity
mulMulmul_gru_8_gru_cell_8_betamul_gru_8_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_8_gru_cell_8_betamul_gru_8_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Å	
ó
B__inference_dense_2_layer_call_and_return_conditional_losses_51328

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

w
"__inference_internal_grad_fn_52690
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_50000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_7_readvariableop_resource_0:	¬D
1while_gru_cell_7_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_7_readvariableop_resource:	¬B
/while_gru_cell_7_matmul_readvariableop_resource:	d¬D
1while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_7/MatMul/ReadVariableOp¢(while/gru_cell_7/MatMul_1/ReadVariableOp¢while/gru_cell_7/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_7/ReadVariableOpReadVariableOp*while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_7/unstackUnpack'while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_7/BiasAddBiasAdd!while/gru_cell_7/MatMul:product:0!while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_7/splitSplit)while/gru_cell_7/split/split_dim:output:0!while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_7/MatMul_1MatMulwhile_placeholder_20while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_7/BiasAdd_1BiasAdd#while/gru_cell_7/MatMul_1:product:0!while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_7/split_1SplitV#while/gru_cell_7/BiasAdd_1:output:0while/gru_cell_7/Const:output:0+while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_7/addAddV2while/gru_cell_7/split:output:0!while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_7/SigmoidSigmoidwhile/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_1AddV2while/gru_cell_7/split:output:1!while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_1Sigmoidwhile/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mulMulwhile/gru_cell_7/Sigmoid_1:y:0!while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_2AddV2while/gru_cell_7/split:output:2while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/mul_1Mulwhile/gru_cell_7/beta:output:0while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_2Sigmoidwhile/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_2Mulwhile/gru_cell_7/add_2:z:0while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/IdentityIdentitywhile/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_7/IdentityN	IdentityNwhile/gru_cell_7/mul_2:z:0while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50050*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_3Mulwhile/gru_cell_7/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/subSubwhile/gru_cell_7/sub/x:output:0while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_4Mulwhile/gru_cell_7/sub:z:0#while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_3AddV2while/gru_cell_7/mul_3:z:0while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_7/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_7/MatMul/ReadVariableOp)^while/gru_cell_7/MatMul_1/ReadVariableOp ^while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_7_matmul_1_readvariableop_resource3while_gru_cell_7_matmul_1_readvariableop_resource_0"d
/while_gru_cell_7_matmul_readvariableop_resource1while_gru_cell_7_matmul_readvariableop_resource_0"V
(while_gru_cell_7_readvariableop_resource*while_gru_cell_7_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_7/MatMul/ReadVariableOp&while/gru_cell_7/MatMul/ReadVariableOp2T
(while/gru_cell_7/MatMul_1/ReadVariableOp(while/gru_cell_7/MatMul_1/ReadVariableOp2B
while/gru_cell_7/ReadVariableOpwhile/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý

"__inference_internal_grad_fn_52222
result_grads_0
result_grads_1
mul_gru_6_gru_cell_6_beta
mul_gru_6_gru_cell_6_add_2
identity
mulMulmul_gru_6_gru_cell_6_betamul_gru_6_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_6_gru_cell_6_betamul_gru_6_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_49287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_49287___redundant_placeholder03
/while_while_cond_49287___redundant_placeholder13
/while_while_cond_49287___redundant_placeholder23
/while_while_cond_49287___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ËQ

@__inference_gru_8_layer_call_and_return_conditional_losses_47288

inputs5
"gru_cell_8_readvariableop_resource:	¬<
)gru_cell_8_matmul_readvariableop_resource:	d¬>
+gru_cell_8_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_8/MatMul/ReadVariableOp¢"gru_cell_8/MatMul_1/ReadVariableOp¢gru_cell_8/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_8/ReadVariableOpReadVariableOp"gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_8/unstackUnpack!gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_8/MatMul/ReadVariableOpReadVariableOp)gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMulMatMulstrided_slice_2:output:0(gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAddBiasAddgru_cell_8/MatMul:product:0gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_8/splitSplit#gru_cell_8/split/split_dim:output:0gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMul_1MatMulzeros:output:0*gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAdd_1BiasAddgru_cell_8/MatMul_1:product:0gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_8/split_1SplitVgru_cell_8/BiasAdd_1:output:0gru_cell_8/Const:output:0%gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_8/addAddV2gru_cell_8/split:output:0gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_8/SigmoidSigmoidgru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_8/add_1AddV2gru_cell_8/split:output:1gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_1Sigmoidgru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_8/mulMulgru_cell_8/Sigmoid_1:y:0gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_8/add_2AddV2gru_cell_8/split:output:2gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_8/mul_1Mulgru_cell_8/beta:output:0gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_2Sigmoidgru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_8/mul_2Mulgru_cell_8/add_2:z:0gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/IdentityIdentitygru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_8/IdentityN	IdentityNgru_cell_8/mul_2:z:0gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47176*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_8/mul_3Mulgru_cell_8/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_8/subSubgru_cell_8/sub/x:output:0gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_8/mul_4Mulgru_cell_8/sub:z:0gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_8/add_3AddV2gru_cell_8/mul_3:z:0gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_8_readvariableop_resource)gru_cell_8_matmul_readvariableop_resource+gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_47192*
condR
while_cond_47191*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_8/MatMul/ReadVariableOp#^gru_cell_8/MatMul_1/ReadVariableOp^gru_cell_8/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_8/MatMul/ReadVariableOp gru_cell_8/MatMul/ReadVariableOp2H
"gru_cell_8/MatMul_1/ReadVariableOp"gru_cell_8/MatMul_1/ReadVariableOp26
gru_cell_8/ReadVariableOpgru_cell_8/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ËQ

@__inference_gru_8_layer_call_and_return_conditional_losses_51309

inputs5
"gru_cell_8_readvariableop_resource:	¬<
)gru_cell_8_matmul_readvariableop_resource:	d¬>
+gru_cell_8_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_8/MatMul/ReadVariableOp¢"gru_cell_8/MatMul_1/ReadVariableOp¢gru_cell_8/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_8/ReadVariableOpReadVariableOp"gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_8/unstackUnpack!gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_8/MatMul/ReadVariableOpReadVariableOp)gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMulMatMulstrided_slice_2:output:0(gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAddBiasAddgru_cell_8/MatMul:product:0gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_8/splitSplit#gru_cell_8/split/split_dim:output:0gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMul_1MatMulzeros:output:0*gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAdd_1BiasAddgru_cell_8/MatMul_1:product:0gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_8/split_1SplitVgru_cell_8/BiasAdd_1:output:0gru_cell_8/Const:output:0%gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_8/addAddV2gru_cell_8/split:output:0gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_8/SigmoidSigmoidgru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_8/add_1AddV2gru_cell_8/split:output:1gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_1Sigmoidgru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_8/mulMulgru_cell_8/Sigmoid_1:y:0gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_8/add_2AddV2gru_cell_8/split:output:2gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_8/mul_1Mulgru_cell_8/beta:output:0gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_2Sigmoidgru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_8/mul_2Mulgru_cell_8/add_2:z:0gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/IdentityIdentitygru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_8/IdentityN	IdentityNgru_cell_8/mul_2:z:0gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51197*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_8/mul_3Mulgru_cell_8/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_8/subSubgru_cell_8/sub/x:output:0gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_8/mul_4Mulgru_cell_8/sub:z:0gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_8/add_3AddV2gru_cell_8/mul_3:z:0gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_8_readvariableop_resource)gru_cell_8_matmul_readvariableop_resource+gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_51213*
condR
while_cond_51212*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_8/MatMul/ReadVariableOp#^gru_cell_8/MatMul_1/ReadVariableOp^gru_cell_8/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_8/MatMul/ReadVariableOp gru_cell_8/MatMul/ReadVariableOp2H
"gru_cell_8/MatMul_1/ReadVariableOp"gru_cell_8/MatMul_1/ReadVariableOp26
gru_cell_8/ReadVariableOpgru_cell_8/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ð3
û
@__inference_gru_8_layer_call_and_return_conditional_losses_46569

inputs#
gru_cell_8_46493:	¬#
gru_cell_8_46495:	d¬#
gru_cell_8_46497:	d¬
identity¢"gru_cell_8/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÀ
"gru_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_8_46493gru_cell_8_46495gru_cell_8_46497*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46492n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_8_46493gru_cell_8_46495gru_cell_8_46497*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46505*
condR
while_cond_46504*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
NoOpNoOp#^gru_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2H
"gru_cell_8/StatefulPartitionedCall"gru_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
!
Ù
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46642

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-46628*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdS
mul_3MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates

w
"__inference_internal_grad_fn_52888
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_49455
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_6_readvariableop_resource_0:	¬D
1while_gru_cell_6_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_6_readvariableop_resource:	¬B
/while_gru_cell_6_matmul_readvariableop_resource:	¬D
1while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_6/MatMul/ReadVariableOp¢(while/gru_cell_6/MatMul_1/ReadVariableOp¢while/gru_cell_6/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_6/ReadVariableOpReadVariableOp*while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_6/unstackUnpack'while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_6/BiasAddBiasAdd!while/gru_cell_6/MatMul:product:0!while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_6/splitSplit)while/gru_cell_6/split/split_dim:output:0!while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_6/MatMul_1MatMulwhile_placeholder_20while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_6/BiasAdd_1BiasAdd#while/gru_cell_6/MatMul_1:product:0!while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_6/split_1SplitV#while/gru_cell_6/BiasAdd_1:output:0while/gru_cell_6/Const:output:0+while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_6/addAddV2while/gru_cell_6/split:output:0!while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_6/SigmoidSigmoidwhile/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_1AddV2while/gru_cell_6/split:output:1!while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_1Sigmoidwhile/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mulMulwhile/gru_cell_6/Sigmoid_1:y:0!while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_2AddV2while/gru_cell_6/split:output:2while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/mul_1Mulwhile/gru_cell_6/beta:output:0while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_2Sigmoidwhile/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_2Mulwhile/gru_cell_6/add_2:z:0while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/IdentityIdentitywhile/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_6/IdentityN	IdentityNwhile/gru_cell_6/mul_2:z:0while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49505*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_3Mulwhile/gru_cell_6/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/subSubwhile/gru_cell_6/sub/x:output:0while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_4Mulwhile/gru_cell_6/sub:z:0#while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_3AddV2while/gru_cell_6/mul_3:z:0while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_6/MatMul/ReadVariableOp)^while/gru_cell_6/MatMul_1/ReadVariableOp ^while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_6_matmul_1_readvariableop_resource3while_gru_cell_6_matmul_1_readvariableop_resource_0"d
/while_gru_cell_6_matmul_readvariableop_resource1while_gru_cell_6_matmul_readvariableop_resource_0"V
(while_gru_cell_6_readvariableop_resource*while_gru_cell_6_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_6/MatMul/ReadVariableOp&while/gru_cell_6/MatMul/ReadVariableOp2T
(while/gru_cell_6/MatMul_1/ReadVariableOp(while/gru_cell_6/MatMul_1/ReadVariableOp2B
while/gru_cell_6/ReadVariableOpwhile/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ËQ

@__inference_gru_8_layer_call_and_return_conditional_losses_47528

inputs5
"gru_cell_8_readvariableop_resource:	¬<
)gru_cell_8_matmul_readvariableop_resource:	d¬>
+gru_cell_8_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_8/MatMul/ReadVariableOp¢"gru_cell_8/MatMul_1/ReadVariableOp¢gru_cell_8/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_8/ReadVariableOpReadVariableOp"gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_8/unstackUnpack!gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_8/MatMul/ReadVariableOpReadVariableOp)gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMulMatMulstrided_slice_2:output:0(gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAddBiasAddgru_cell_8/MatMul:product:0gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_8/splitSplit#gru_cell_8/split/split_dim:output:0gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMul_1MatMulzeros:output:0*gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAdd_1BiasAddgru_cell_8/MatMul_1:product:0gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_8/split_1SplitVgru_cell_8/BiasAdd_1:output:0gru_cell_8/Const:output:0%gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_8/addAddV2gru_cell_8/split:output:0gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_8/SigmoidSigmoidgru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_8/add_1AddV2gru_cell_8/split:output:1gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_1Sigmoidgru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_8/mulMulgru_cell_8/Sigmoid_1:y:0gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_8/add_2AddV2gru_cell_8/split:output:2gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_8/mul_1Mulgru_cell_8/beta:output:0gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_2Sigmoidgru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_8/mul_2Mulgru_cell_8/add_2:z:0gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/IdentityIdentitygru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_8/IdentityN	IdentityNgru_cell_8/mul_2:z:0gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47416*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_8/mul_3Mulgru_cell_8/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_8/subSubgru_cell_8/sub/x:output:0gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_8/mul_4Mulgru_cell_8/sub:z:0gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_8/add_3AddV2gru_cell_8/mul_3:z:0gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_8_readvariableop_resource)gru_cell_8_matmul_readvariableop_resource+gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_47432*
condR
while_cond_47431*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_8/MatMul/ReadVariableOp#^gru_cell_8/MatMul_1/ReadVariableOp^gru_cell_8/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_8/MatMul/ReadVariableOp gru_cell_8/MatMul/ReadVariableOp2H
"gru_cell_8/MatMul_1/ReadVariableOp"gru_cell_8/MatMul_1/ReadVariableOp26
gru_cell_8/ReadVariableOpgru_cell_8/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
B
þ
while_body_47810
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_6_readvariableop_resource_0:	¬D
1while_gru_cell_6_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_6_readvariableop_resource:	¬B
/while_gru_cell_6_matmul_readvariableop_resource:	¬D
1while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_6/MatMul/ReadVariableOp¢(while/gru_cell_6/MatMul_1/ReadVariableOp¢while/gru_cell_6/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_6/ReadVariableOpReadVariableOp*while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_6/unstackUnpack'while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_6/BiasAddBiasAdd!while/gru_cell_6/MatMul:product:0!while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_6/splitSplit)while/gru_cell_6/split/split_dim:output:0!while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_6/MatMul_1MatMulwhile_placeholder_20while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_6/BiasAdd_1BiasAdd#while/gru_cell_6/MatMul_1:product:0!while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_6/split_1SplitV#while/gru_cell_6/BiasAdd_1:output:0while/gru_cell_6/Const:output:0+while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_6/addAddV2while/gru_cell_6/split:output:0!while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_6/SigmoidSigmoidwhile/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_1AddV2while/gru_cell_6/split:output:1!while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_1Sigmoidwhile/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mulMulwhile/gru_cell_6/Sigmoid_1:y:0!while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_2AddV2while/gru_cell_6/split:output:2while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/mul_1Mulwhile/gru_cell_6/beta:output:0while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_2Sigmoidwhile/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_2Mulwhile/gru_cell_6/add_2:z:0while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/IdentityIdentitywhile/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_6/IdentityN	IdentityNwhile/gru_cell_6/mul_2:z:0while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47860*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_3Mulwhile/gru_cell_6/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/subSubwhile/gru_cell_6/sub/x:output:0while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_4Mulwhile/gru_cell_6/sub:z:0#while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_3AddV2while/gru_cell_6/mul_3:z:0while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_6/MatMul/ReadVariableOp)^while/gru_cell_6/MatMul_1/ReadVariableOp ^while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_6_matmul_1_readvariableop_resource3while_gru_cell_6_matmul_1_readvariableop_resource_0"d
/while_gru_cell_6_matmul_readvariableop_resource1while_gru_cell_6_matmul_readvariableop_resource_0"V
(while_gru_cell_6_readvariableop_resource*while_gru_cell_6_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_6/MatMul/ReadVariableOp&while/gru_cell_6/MatMul/ReadVariableOp2T
(while/gru_cell_6/MatMul_1/ReadVariableOp(while/gru_cell_6/MatMul_1/ReadVariableOp2B
while/gru_cell_6/ReadVariableOpwhile/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
B
þ
while_body_50712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_8_readvariableop_resource_0:	¬D
1while_gru_cell_8_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_8_readvariableop_resource:	¬B
/while_gru_cell_8_matmul_readvariableop_resource:	d¬D
1while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_8/MatMul/ReadVariableOp¢(while/gru_cell_8/MatMul_1/ReadVariableOp¢while/gru_cell_8/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_8/ReadVariableOpReadVariableOp*while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_8/unstackUnpack'while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_8/BiasAddBiasAdd!while/gru_cell_8/MatMul:product:0!while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_8/splitSplit)while/gru_cell_8/split/split_dim:output:0!while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_8/MatMul_1MatMulwhile_placeholder_20while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_8/BiasAdd_1BiasAdd#while/gru_cell_8/MatMul_1:product:0!while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_8/split_1SplitV#while/gru_cell_8/BiasAdd_1:output:0while/gru_cell_8/Const:output:0+while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_8/addAddV2while/gru_cell_8/split:output:0!while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_8/SigmoidSigmoidwhile/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_1AddV2while/gru_cell_8/split:output:1!while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_1Sigmoidwhile/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mulMulwhile/gru_cell_8/Sigmoid_1:y:0!while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_2AddV2while/gru_cell_8/split:output:2while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/mul_1Mulwhile/gru_cell_8/beta:output:0while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_2Sigmoidwhile/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_2Mulwhile/gru_cell_8/add_2:z:0while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/IdentityIdentitywhile/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_8/IdentityN	IdentityNwhile/gru_cell_8/mul_2:z:0while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50762*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_3Mulwhile/gru_cell_8/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/subSubwhile/gru_cell_8/sub/x:output:0while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_4Mulwhile/gru_cell_8/sub:z:0#while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_3AddV2while/gru_cell_8/mul_3:z:0while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_8/MatMul/ReadVariableOp)^while/gru_cell_8/MatMul_1/ReadVariableOp ^while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_8_matmul_1_readvariableop_resource3while_gru_cell_8_matmul_1_readvariableop_resource_0"d
/while_gru_cell_8_matmul_readvariableop_resource1while_gru_cell_8_matmul_readvariableop_resource_0"V
(while_gru_cell_8_readvariableop_resource*while_gru_cell_8_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_8/MatMul/ReadVariableOp&while/gru_cell_8/MatMul/ReadVariableOp2T
(while/gru_cell_8/MatMul_1/ReadVariableOp(while/gru_cell_8/MatMul_1/ReadVariableOp2B
while/gru_cell_8/ReadVariableOpwhile/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý
¶
%__inference_gru_6_layer_call_fn_49206

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_46940s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ø

"__inference_internal_grad_fn_52474
result_grads_0
result_grads_1
mul_gru_cell_6_beta
mul_gru_cell_6_add_2
identityx
mulMulmul_gru_cell_6_betamul_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_6_betamul_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý

"__inference_internal_grad_fn_51916
result_grads_0
result_grads_1
mul_while_gru_cell_6_beta
mul_while_gru_cell_6_add_2
identity
mulMulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Æ
G__inference_sequential_2_layer_call_and_return_conditional_losses_47974

inputs
gru_6_47947:	¬
gru_6_47949:	¬
gru_6_47951:	d¬
gru_7_47954:	¬
gru_7_47956:	d¬
gru_7_47958:	d¬
gru_8_47961:	¬
gru_8_47963:	d¬
gru_8_47965:	d¬
dense_2_47968:d
dense_2_47970:
identity¢dense_2/StatefulPartitionedCall¢gru_6/StatefulPartitionedCall¢gru_7/StatefulPartitionedCall¢gru_8/StatefulPartitionedCallô
gru_6/StatefulPartitionedCallStatefulPartitionedCallinputsgru_6_47947gru_6_47949gru_6_47951*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_47906
gru_7/StatefulPartitionedCallStatefulPartitionedCall&gru_6/StatefulPartitionedCall:output:0gru_7_47954gru_7_47956gru_7_47958*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_47717
gru_8/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0gru_8_47961gru_8_47963gru_8_47965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_47528
dense_2/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0dense_2_47968dense_2_47970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_47306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_2/StatefulPartitionedCall^gru_6/StatefulPartitionedCall^gru_7/StatefulPartitionedCall^gru_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
gru_6/StatefulPartitionedCallgru_6/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ð
¿
"__inference_internal_grad_fn_51880
result_grads_0
result_grads_10
,mul_sequential_2_gru_8_while_gru_cell_8_beta1
-mul_sequential_2_gru_8_while_gru_cell_8_add_2
identityª
mulMul,mul_sequential_2_gru_8_while_gru_cell_8_beta-mul_sequential_2_gru_8_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul,mul_sequential_2_gru_8_while_gru_cell_8_beta-mul_sequential_2_gru_8_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý

"__inference_internal_grad_fn_51952
result_grads_0
result_grads_1
mul_while_gru_cell_7_beta
mul_while_gru_cell_7_add_2
identity
mulMulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ð
¿
"__inference_internal_grad_fn_51862
result_grads_0
result_grads_10
,mul_sequential_2_gru_7_while_gru_cell_7_beta1
-mul_sequential_2_gru_7_while_gru_cell_7_add_2
identityª
mulMul,mul_sequential_2_gru_7_while_gru_cell_7_beta-mul_sequential_2_gru_7_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul,mul_sequential_2_gru_7_while_gru_cell_7_beta-mul_sequential_2_gru_7_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¢
¥
"__inference_internal_grad_fn_52276
result_grads_0
result_grads_1#
mul_gru_6_while_gru_cell_6_beta$
 mul_gru_6_while_gru_cell_6_add_2
identity
mulMulmul_gru_6_while_gru_cell_6_beta mul_gru_6_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_6_while_gru_cell_6_beta mul_gru_6_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
÷

gru_7_while_cond_48878(
$gru_7_while_gru_7_while_loop_counter.
*gru_7_while_gru_7_while_maximum_iterations
gru_7_while_placeholder
gru_7_while_placeholder_1
gru_7_while_placeholder_2*
&gru_7_while_less_gru_7_strided_slice_1?
;gru_7_while_gru_7_while_cond_48878___redundant_placeholder0?
;gru_7_while_gru_7_while_cond_48878___redundant_placeholder1?
;gru_7_while_gru_7_while_cond_48878___redundant_placeholder2?
;gru_7_while_gru_7_while_cond_48878___redundant_placeholder3
gru_7_while_identity
z
gru_7/while/LessLessgru_7_while_placeholder&gru_7_while_less_gru_7_strided_slice_1*
T0*
_output_shapes
: W
gru_7/while/IdentityIdentitygru_7/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_7_while_identitygru_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:

w
"__inference_internal_grad_fn_52330
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ø

"__inference_internal_grad_fn_52654
result_grads_0
result_grads_1
mul_gru_cell_7_beta
mul_gru_cell_7_add_2
identityx
mulMulmul_gru_cell_7_betamul_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_7_betamul_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
õ
¶
%__inference_gru_8_layer_call_fn_50641

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_47528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ø

"__inference_internal_grad_fn_52726
result_grads_0
result_grads_1
mul_gru_cell_8_beta
mul_gru_cell_8_add_2
identityx
mulMulmul_gru_cell_8_betamul_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_8_betamul_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
×

#sequential_2_gru_8_while_cond_45608B
>sequential_2_gru_8_while_sequential_2_gru_8_while_loop_counterH
Dsequential_2_gru_8_while_sequential_2_gru_8_while_maximum_iterations(
$sequential_2_gru_8_while_placeholder*
&sequential_2_gru_8_while_placeholder_1*
&sequential_2_gru_8_while_placeholder_2D
@sequential_2_gru_8_while_less_sequential_2_gru_8_strided_slice_1Y
Usequential_2_gru_8_while_sequential_2_gru_8_while_cond_45608___redundant_placeholder0Y
Usequential_2_gru_8_while_sequential_2_gru_8_while_cond_45608___redundant_placeholder1Y
Usequential_2_gru_8_while_sequential_2_gru_8_while_cond_45608___redundant_placeholder2Y
Usequential_2_gru_8_while_sequential_2_gru_8_while_cond_45608___redundant_placeholder3%
!sequential_2_gru_8_while_identity
®
sequential_2/gru_8/while/LessLess$sequential_2_gru_8_while_placeholder@sequential_2_gru_8_while_less_sequential_2_gru_8_strided_slice_1*
T0*
_output_shapes
: q
!sequential_2/gru_8/while/IdentityIdentity!sequential_2/gru_8/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_2_gru_8_while_identity*sequential_2/gru_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ø

"__inference_internal_grad_fn_52078
result_grads_0
result_grads_1
mul_gru_cell_8_beta
mul_gru_cell_8_add_2
identityx
mulMulmul_gru_cell_8_betamul_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_8_betamul_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_47431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47431___redundant_placeholder03
/while_while_cond_47431___redundant_placeholder13
/while_while_cond_47431___redundant_placeholder23
/while_while_cond_47431___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ý

"__inference_internal_grad_fn_52492
result_grads_0
result_grads_1
mul_while_gru_cell_6_beta
mul_while_gru_cell_6_add_2
identity
mulMulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_46504
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46504___redundant_placeholder03
/while_while_cond_46504___redundant_placeholder13
/while_while_cond_46504___redundant_placeholder23
/while_while_cond_46504___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
å
§
while_body_46342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_7_46364_0:	¬+
while_gru_cell_7_46366_0:	d¬+
while_gru_cell_7_46368_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_7_46364:	¬)
while_gru_cell_7_46366:	d¬)
while_gru_cell_7_46368:	d¬¢(while/gru_cell_7/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0û
(while/gru_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_7_46364_0while_gru_cell_7_46366_0while_gru_cell_7_46368_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46290Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_7_46364while_gru_cell_7_46364_0"2
while_gru_cell_7_46366while_gru_cell_7_46366_0"2
while_gru_cell_7_46368while_gru_cell_7_46368_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/gru_cell_7/StatefulPartitionedCall(while/gru_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
B
þ
while_body_47621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_7_readvariableop_resource_0:	¬D
1while_gru_cell_7_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_7_readvariableop_resource:	¬B
/while_gru_cell_7_matmul_readvariableop_resource:	d¬D
1while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_7/MatMul/ReadVariableOp¢(while/gru_cell_7/MatMul_1/ReadVariableOp¢while/gru_cell_7/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_7/ReadVariableOpReadVariableOp*while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_7/unstackUnpack'while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_7/BiasAddBiasAdd!while/gru_cell_7/MatMul:product:0!while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_7/splitSplit)while/gru_cell_7/split/split_dim:output:0!while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_7/MatMul_1MatMulwhile_placeholder_20while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_7/BiasAdd_1BiasAdd#while/gru_cell_7/MatMul_1:product:0!while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_7/split_1SplitV#while/gru_cell_7/BiasAdd_1:output:0while/gru_cell_7/Const:output:0+while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_7/addAddV2while/gru_cell_7/split:output:0!while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_7/SigmoidSigmoidwhile/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_1AddV2while/gru_cell_7/split:output:1!while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_1Sigmoidwhile/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mulMulwhile/gru_cell_7/Sigmoid_1:y:0!while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_2AddV2while/gru_cell_7/split:output:2while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/mul_1Mulwhile/gru_cell_7/beta:output:0while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_2Sigmoidwhile/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_2Mulwhile/gru_cell_7/add_2:z:0while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/IdentityIdentitywhile/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_7/IdentityN	IdentityNwhile/gru_cell_7/mul_2:z:0while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47671*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_3Mulwhile/gru_cell_7/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/subSubwhile/gru_cell_7/sub/x:output:0while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_4Mulwhile/gru_cell_7/sub:z:0#while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_3AddV2while/gru_cell_7/mul_3:z:0while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_7/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_7/MatMul/ReadVariableOp)^while/gru_cell_7/MatMul_1/ReadVariableOp ^while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_7_matmul_1_readvariableop_resource3while_gru_cell_7_matmul_1_readvariableop_resource_0"d
/while_gru_cell_7_matmul_readvariableop_resource1while_gru_cell_7_matmul_readvariableop_resource_0"V
(while_gru_cell_7_readvariableop_resource*while_gru_cell_7_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_7/MatMul/ReadVariableOp&while/gru_cell_7/MatMul/ReadVariableOp2T
(while/gru_cell_7/MatMul_1/ReadVariableOp(while/gru_cell_7/MatMul_1/ReadVariableOp2B
while/gru_cell_7/ReadVariableOpwhile/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý

"__inference_internal_grad_fn_52114
result_grads_0
result_grads_1
mul_gru_6_gru_cell_6_beta
mul_gru_6_gru_cell_6_add_2
identity
mulMulmul_gru_6_gru_cell_6_betamul_gru_6_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_6_gru_cell_6_betamul_gru_6_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
÷

gru_6_while_cond_48715(
$gru_6_while_gru_6_while_loop_counter.
*gru_6_while_gru_6_while_maximum_iterations
gru_6_while_placeholder
gru_6_while_placeholder_1
gru_6_while_placeholder_2*
&gru_6_while_less_gru_6_strided_slice_1?
;gru_6_while_gru_6_while_cond_48715___redundant_placeholder0?
;gru_6_while_gru_6_while_cond_48715___redundant_placeholder1?
;gru_6_while_gru_6_while_cond_48715___redundant_placeholder2?
;gru_6_while_gru_6_while_cond_48715___redundant_placeholder3
gru_6_while_identity
z
gru_6/while/LessLessgru_6_while_placeholder&gru_6_while_less_gru_6_strided_slice_1*
T0*
_output_shapes
: W
gru_6/while/IdentityIdentitygru_6/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_6_while_identitygru_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
R

@__inference_gru_6_layer_call_and_return_conditional_losses_49384
inputs_05
"gru_cell_6_readvariableop_resource:	¬<
)gru_cell_6_matmul_readvariableop_resource:	¬>
+gru_cell_6_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_6/MatMul/ReadVariableOp¢"gru_cell_6/MatMul_1/ReadVariableOp¢gru_cell_6/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_6/ReadVariableOpReadVariableOp"gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_6/unstackUnpack!gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_6/MatMul/ReadVariableOpReadVariableOp)gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_6/MatMulMatMulstrided_slice_2:output:0(gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAddBiasAddgru_cell_6/MatMul:product:0gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_6/splitSplit#gru_cell_6/split/split_dim:output:0gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_6/MatMul_1MatMulzeros:output:0*gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAdd_1BiasAddgru_cell_6/MatMul_1:product:0gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_6/split_1SplitVgru_cell_6/BiasAdd_1:output:0gru_cell_6/Const:output:0%gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_6/addAddV2gru_cell_6/split:output:0gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_6/SigmoidSigmoidgru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_6/add_1AddV2gru_cell_6/split:output:1gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_1Sigmoidgru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_6/mulMulgru_cell_6/Sigmoid_1:y:0gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_6/add_2AddV2gru_cell_6/split:output:2gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_6/mul_1Mulgru_cell_6/beta:output:0gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_2Sigmoidgru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_6/mul_2Mulgru_cell_6/add_2:z:0gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/IdentityIdentitygru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_6/IdentityN	IdentityNgru_cell_6/mul_2:z:0gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49272*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_6/mul_3Mulgru_cell_6/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_6/subSubgru_cell_6/sub/x:output:0gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_6/mul_4Mulgru_cell_6/sub:z:0gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_6/add_3AddV2gru_cell_6/mul_3:z:0gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_6_readvariableop_resource)gru_cell_6_matmul_readvariableop_resource+gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_49288*
condR
while_cond_49287*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_6/MatMul/ReadVariableOp#^gru_cell_6/MatMul_1/ReadVariableOp^gru_cell_6/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_6/MatMul/ReadVariableOp gru_cell_6/MatMul/ReadVariableOp2H
"gru_cell_6/MatMul_1/ReadVariableOp"gru_cell_6/MatMul_1/ReadVariableOp26
gru_cell_6/ReadVariableOpgru_cell_6/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

w
"__inference_internal_grad_fn_52528
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ô

¬
,__inference_sequential_2_layer_call_fn_48026
gru_6_input
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
	unknown_2:	¬
	unknown_3:	d¬
	unknown_4:	d¬
	unknown_5:	¬
	unknown_6:	d¬
	unknown_7:	d¬
	unknown_8:d
	unknown_9:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallgru_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_47974o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_6_input
Ø

"__inference_internal_grad_fn_52834
result_grads_0
result_grads_1
mul_gru_cell_8_beta
mul_gru_cell_8_add_2
identityx
mulMulmul_gru_cell_8_betamul_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_8_betamul_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
°

Ù
*__inference_gru_cell_8_layer_call_fn_51582

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46492o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
Õ
¥
while_cond_46341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46341___redundant_placeholder03
/while_while_cond_46341___redundant_placeholder13
/while_while_cond_46341___redundant_placeholder23
/while_while_cond_46341___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
æ[
¸
#sequential_2_gru_6_while_body_45283B
>sequential_2_gru_6_while_sequential_2_gru_6_while_loop_counterH
Dsequential_2_gru_6_while_sequential_2_gru_6_while_maximum_iterations(
$sequential_2_gru_6_while_placeholder*
&sequential_2_gru_6_while_placeholder_1*
&sequential_2_gru_6_while_placeholder_2A
=sequential_2_gru_6_while_sequential_2_gru_6_strided_slice_1_0}
ysequential_2_gru_6_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_6_tensorarrayunstack_tensorlistfromtensor_0P
=sequential_2_gru_6_while_gru_cell_6_readvariableop_resource_0:	¬W
Dsequential_2_gru_6_while_gru_cell_6_matmul_readvariableop_resource_0:	¬Y
Fsequential_2_gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬%
!sequential_2_gru_6_while_identity'
#sequential_2_gru_6_while_identity_1'
#sequential_2_gru_6_while_identity_2'
#sequential_2_gru_6_while_identity_3'
#sequential_2_gru_6_while_identity_4?
;sequential_2_gru_6_while_sequential_2_gru_6_strided_slice_1{
wsequential_2_gru_6_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_6_tensorarrayunstack_tensorlistfromtensorN
;sequential_2_gru_6_while_gru_cell_6_readvariableop_resource:	¬U
Bsequential_2_gru_6_while_gru_cell_6_matmul_readvariableop_resource:	¬W
Dsequential_2_gru_6_while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢9sequential_2/gru_6/while/gru_cell_6/MatMul/ReadVariableOp¢;sequential_2/gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp¢2sequential_2/gru_6/while/gru_cell_6/ReadVariableOp
Jsequential_2/gru_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<sequential_2/gru_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_2_gru_6_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_6_tensorarrayunstack_tensorlistfromtensor_0$sequential_2_gru_6_while_placeholderSsequential_2/gru_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0±
2sequential_2/gru_6/while/gru_cell_6/ReadVariableOpReadVariableOp=sequential_2_gru_6_while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0©
+sequential_2/gru_6/while/gru_cell_6/unstackUnpack:sequential_2/gru_6/while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¿
9sequential_2/gru_6/while/gru_cell_6/MatMul/ReadVariableOpReadVariableOpDsequential_2_gru_6_while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0ï
*sequential_2/gru_6/while/gru_cell_6/MatMulMatMulCsequential_2/gru_6/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_2/gru_6/while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Õ
+sequential_2/gru_6/while/gru_cell_6/BiasAddBiasAdd4sequential_2/gru_6/while/gru_cell_6/MatMul:product:04sequential_2/gru_6/while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
3sequential_2/gru_6/while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
)sequential_2/gru_6/while/gru_cell_6/splitSplit<sequential_2/gru_6/while/gru_cell_6/split/split_dim:output:04sequential_2/gru_6/while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÃ
;sequential_2/gru_6/while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOpFsequential_2_gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ö
,sequential_2/gru_6/while/gru_cell_6/MatMul_1MatMul&sequential_2_gru_6_while_placeholder_2Csequential_2/gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ù
-sequential_2/gru_6/while/gru_cell_6/BiasAdd_1BiasAdd6sequential_2/gru_6/while/gru_cell_6/MatMul_1:product:04sequential_2/gru_6/while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
)sequential_2/gru_6/while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ
5sequential_2/gru_6/while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
+sequential_2/gru_6/while/gru_cell_6/split_1SplitV6sequential_2/gru_6/while/gru_cell_6/BiasAdd_1:output:02sequential_2/gru_6/while/gru_cell_6/Const:output:0>sequential_2/gru_6/while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÌ
'sequential_2/gru_6/while/gru_cell_6/addAddV22sequential_2/gru_6/while/gru_cell_6/split:output:04sequential_2/gru_6/while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+sequential_2/gru_6/while/gru_cell_6/SigmoidSigmoid+sequential_2/gru_6/while/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÎ
)sequential_2/gru_6/while/gru_cell_6/add_1AddV22sequential_2/gru_6/while/gru_cell_6/split:output:14sequential_2/gru_6/while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_6/while/gru_cell_6/Sigmoid_1Sigmoid-sequential_2/gru_6/while/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÉ
'sequential_2/gru_6/while/gru_cell_6/mulMul1sequential_2/gru_6/while/gru_cell_6/Sigmoid_1:y:04sequential_2/gru_6/while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ
)sequential_2/gru_6/while/gru_cell_6/add_2AddV22sequential_2/gru_6/while/gru_cell_6/split:output:2+sequential_2/gru_6/while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(sequential_2/gru_6/while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
)sequential_2/gru_6/while/gru_cell_6/mul_1Mul1sequential_2/gru_6/while/gru_cell_6/beta:output:0-sequential_2/gru_6/while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_6/while/gru_cell_6/Sigmoid_2Sigmoid-sequential_2/gru_6/while/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
)sequential_2/gru_6/while/gru_cell_6/mul_2Mul-sequential_2/gru_6/while/gru_cell_6/add_2:z:01sequential_2/gru_6/while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_2/gru_6/while/gru_cell_6/IdentityIdentity-sequential_2/gru_6/while/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_6/while/gru_cell_6/IdentityN	IdentityN-sequential_2/gru_6/while/gru_cell_6/mul_2:z:0-sequential_2/gru_6/while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45333*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd»
)sequential_2/gru_6/while/gru_cell_6/mul_3Mul/sequential_2/gru_6/while/gru_cell_6/Sigmoid:y:0&sequential_2_gru_6_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
)sequential_2/gru_6/while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Å
'sequential_2/gru_6/while/gru_cell_6/subSub2sequential_2/gru_6/while/gru_cell_6/sub/x:output:0/sequential_2/gru_6/while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
)sequential_2/gru_6/while/gru_cell_6/mul_4Mul+sequential_2/gru_6/while/gru_cell_6/sub:z:06sequential_2/gru_6/while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ
)sequential_2/gru_6/while/gru_cell_6/add_3AddV2-sequential_2/gru_6/while/gru_cell_6/mul_3:z:0-sequential_2/gru_6/while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
=sequential_2/gru_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_2_gru_6_while_placeholder_1$sequential_2_gru_6_while_placeholder-sequential_2/gru_6/while/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_2/gru_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_2/gru_6/while/addAddV2$sequential_2_gru_6_while_placeholder'sequential_2/gru_6/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_2/gru_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_2/gru_6/while/add_1AddV2>sequential_2_gru_6_while_sequential_2_gru_6_while_loop_counter)sequential_2/gru_6/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_2/gru_6/while/IdentityIdentity"sequential_2/gru_6/while/add_1:z:0^sequential_2/gru_6/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_2/gru_6/while/Identity_1IdentityDsequential_2_gru_6_while_sequential_2_gru_6_while_maximum_iterations^sequential_2/gru_6/while/NoOp*
T0*
_output_shapes
: 
#sequential_2/gru_6/while/Identity_2Identity sequential_2/gru_6/while/add:z:0^sequential_2/gru_6/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_2/gru_6/while/Identity_3IdentityMsequential_2/gru_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/gru_6/while/NoOp*
T0*
_output_shapes
: :éèÒ°
#sequential_2/gru_6/while/Identity_4Identity-sequential_2/gru_6/while/gru_cell_6/add_3:z:0^sequential_2/gru_6/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_2/gru_6/while/NoOpNoOp:^sequential_2/gru_6/while/gru_cell_6/MatMul/ReadVariableOp<^sequential_2/gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp3^sequential_2/gru_6/while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Dsequential_2_gru_6_while_gru_cell_6_matmul_1_readvariableop_resourceFsequential_2_gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0"
Bsequential_2_gru_6_while_gru_cell_6_matmul_readvariableop_resourceDsequential_2_gru_6_while_gru_cell_6_matmul_readvariableop_resource_0"|
;sequential_2_gru_6_while_gru_cell_6_readvariableop_resource=sequential_2_gru_6_while_gru_cell_6_readvariableop_resource_0"O
!sequential_2_gru_6_while_identity*sequential_2/gru_6/while/Identity:output:0"S
#sequential_2_gru_6_while_identity_1,sequential_2/gru_6/while/Identity_1:output:0"S
#sequential_2_gru_6_while_identity_2,sequential_2/gru_6/while/Identity_2:output:0"S
#sequential_2_gru_6_while_identity_3,sequential_2/gru_6/while/Identity_3:output:0"S
#sequential_2_gru_6_while_identity_4,sequential_2/gru_6/while/Identity_4:output:0"|
;sequential_2_gru_6_while_sequential_2_gru_6_strided_slice_1=sequential_2_gru_6_while_sequential_2_gru_6_strided_slice_1_0"ô
wsequential_2_gru_6_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_6_tensorarrayunstack_tensorlistfromtensorysequential_2_gru_6_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2v
9sequential_2/gru_6/while/gru_cell_6/MatMul/ReadVariableOp9sequential_2/gru_6/while/gru_cell_6/MatMul/ReadVariableOp2z
;sequential_2/gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp;sequential_2/gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp2h
2sequential_2/gru_6/while/gru_cell_6/ReadVariableOp2sequential_2/gru_6/while/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_49788
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_49788___redundant_placeholder03
/while_while_cond_49788___redundant_placeholder13
/while_while_cond_49788___redundant_placeholder23
/while_while_cond_49788___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
B
þ
while_body_50879
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_8_readvariableop_resource_0:	¬D
1while_gru_cell_8_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_8_readvariableop_resource:	¬B
/while_gru_cell_8_matmul_readvariableop_resource:	d¬D
1while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_8/MatMul/ReadVariableOp¢(while/gru_cell_8/MatMul_1/ReadVariableOp¢while/gru_cell_8/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_8/ReadVariableOpReadVariableOp*while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_8/unstackUnpack'while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_8/BiasAddBiasAdd!while/gru_cell_8/MatMul:product:0!while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_8/splitSplit)while/gru_cell_8/split/split_dim:output:0!while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_8/MatMul_1MatMulwhile_placeholder_20while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_8/BiasAdd_1BiasAdd#while/gru_cell_8/MatMul_1:product:0!while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_8/split_1SplitV#while/gru_cell_8/BiasAdd_1:output:0while/gru_cell_8/Const:output:0+while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_8/addAddV2while/gru_cell_8/split:output:0!while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_8/SigmoidSigmoidwhile/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_1AddV2while/gru_cell_8/split:output:1!while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_1Sigmoidwhile/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mulMulwhile/gru_cell_8/Sigmoid_1:y:0!while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_2AddV2while/gru_cell_8/split:output:2while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/mul_1Mulwhile/gru_cell_8/beta:output:0while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_2Sigmoidwhile/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_2Mulwhile/gru_cell_8/add_2:z:0while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/IdentityIdentitywhile/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_8/IdentityN	IdentityNwhile/gru_cell_8/mul_2:z:0while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50929*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_3Mulwhile/gru_cell_8/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/subSubwhile/gru_cell_8/sub/x:output:0while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_4Mulwhile/gru_cell_8/sub:z:0#while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_3AddV2while/gru_cell_8/mul_3:z:0while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_8/MatMul/ReadVariableOp)^while/gru_cell_8/MatMul_1/ReadVariableOp ^while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_8_matmul_1_readvariableop_resource3while_gru_cell_8_matmul_1_readvariableop_resource_0"d
/while_gru_cell_8_matmul_readvariableop_resource1while_gru_cell_8_matmul_readvariableop_resource_0"V
(while_gru_cell_8_readvariableop_resource*while_gru_cell_8_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_8/MatMul/ReadVariableOp&while/gru_cell_8/MatMul/ReadVariableOp2T
(while/gru_cell_8/MatMul_1/ReadVariableOp(while/gru_cell_8/MatMul_1/ReadVariableOp2B
while/gru_cell_8/ReadVariableOpwhile/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
¢
¥
"__inference_internal_grad_fn_52294
result_grads_0
result_grads_1#
mul_gru_7_while_gru_cell_7_beta$
 mul_gru_7_while_gru_cell_7_add_2
identity
mulMulmul_gru_7_while_gru_cell_7_beta mul_gru_7_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_7_while_gru_cell_7_beta mul_gru_7_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_49621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_49621___redundant_placeholder03
/while_while_cond_49621___redundant_placeholder13
/while_while_cond_49621___redundant_placeholder23
/while_while_cond_49621___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:

Æ
G__inference_sequential_2_layer_call_and_return_conditional_losses_47313

inputs
gru_6_46941:	¬
gru_6_46943:	¬
gru_6_46945:	d¬
gru_7_47115:	¬
gru_7_47117:	d¬
gru_7_47119:	d¬
gru_8_47289:	¬
gru_8_47291:	d¬
gru_8_47293:	d¬
dense_2_47307:d
dense_2_47309:
identity¢dense_2/StatefulPartitionedCall¢gru_6/StatefulPartitionedCall¢gru_7/StatefulPartitionedCall¢gru_8/StatefulPartitionedCallô
gru_6/StatefulPartitionedCallStatefulPartitionedCallinputsgru_6_46941gru_6_46943gru_6_46945*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_46940
gru_7/StatefulPartitionedCallStatefulPartitionedCall&gru_6/StatefulPartitionedCall:output:0gru_7_47115gru_7_47117gru_7_47119*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_47114
gru_8/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0gru_8_47289gru_8_47291gru_8_47293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_47288
dense_2/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0dense_2_47307dense_2_47309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_47306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_2/StatefulPartitionedCall^gru_6/StatefulPartitionedCall^gru_7/StatefulPartitionedCall^gru_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
gru_6/StatefulPartitionedCallgru_6/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
R

@__inference_gru_7_layer_call_and_return_conditional_losses_50096
inputs_05
"gru_cell_7_readvariableop_resource:	¬<
)gru_cell_7_matmul_readvariableop_resource:	d¬>
+gru_cell_7_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_7/MatMul/ReadVariableOp¢"gru_cell_7/MatMul_1/ReadVariableOp¢gru_cell_7/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_7/ReadVariableOpReadVariableOp"gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_7/unstackUnpack!gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_7/MatMul/ReadVariableOpReadVariableOp)gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMulMatMulstrided_slice_2:output:0(gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAddBiasAddgru_cell_7/MatMul:product:0gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_7/splitSplit#gru_cell_7/split/split_dim:output:0gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMul_1MatMulzeros:output:0*gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAdd_1BiasAddgru_cell_7/MatMul_1:product:0gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_7/split_1SplitVgru_cell_7/BiasAdd_1:output:0gru_cell_7/Const:output:0%gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_7/addAddV2gru_cell_7/split:output:0gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_7/SigmoidSigmoidgru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_7/add_1AddV2gru_cell_7/split:output:1gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_1Sigmoidgru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_7/mulMulgru_cell_7/Sigmoid_1:y:0gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_7/add_2AddV2gru_cell_7/split:output:2gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_7/mul_1Mulgru_cell_7/beta:output:0gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_2Sigmoidgru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_7/mul_2Mulgru_cell_7/add_2:z:0gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/IdentityIdentitygru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_7/IdentityN	IdentityNgru_cell_7/mul_2:z:0gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49984*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_7/mul_3Mulgru_cell_7/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_7/subSubgru_cell_7/sub/x:output:0gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_7/mul_4Mulgru_cell_7/sub:z:0gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_7/add_3AddV2gru_cell_7/mul_3:z:0gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_7_readvariableop_resource)gru_cell_7_matmul_readvariableop_resource+gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_50000*
condR
while_cond_49999*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_7/MatMul/ReadVariableOp#^gru_cell_7/MatMul_1/ReadVariableOp^gru_cell_7/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_7/MatMul/ReadVariableOp gru_cell_7/MatMul/ReadVariableOp2H
"gru_cell_7/MatMul_1/ReadVariableOp"gru_cell_7/MatMul_1/ReadVariableOp26
gru_cell_7/ReadVariableOpgru_cell_7/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
ð3
û
@__inference_gru_8_layer_call_and_return_conditional_losses_46758

inputs#
gru_cell_8_46682:	¬#
gru_cell_8_46684:	d¬#
gru_cell_8_46686:	d¬
identity¢"gru_cell_8/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÀ
"gru_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_8_46682gru_cell_8_46684gru_cell_8_46686*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46642n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_8_46682gru_cell_8_46684gru_cell_8_46686*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46694*
condR
while_cond_46693*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
NoOpNoOp#^gru_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2H
"gru_cell_8/StatefulPartitionedCall"gru_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Õ
¥
while_cond_47017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47017___redundant_placeholder03
/while_while_cond_47017___redundant_placeholder13
/while_while_cond_47017___redundant_placeholder23
/while_while_cond_47017___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
¢
¥
"__inference_internal_grad_fn_52186
result_grads_0
result_grads_1#
mul_gru_7_while_gru_cell_7_beta$
 mul_gru_7_while_gru_cell_7_add_2
identity
mulMulmul_gru_7_while_gru_cell_7_beta mul_gru_7_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_7_while_gru_cell_7_beta mul_gru_7_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ø

"__inference_internal_grad_fn_52366
result_grads_0
result_grads_1
mul_gru_cell_6_beta
mul_gru_cell_6_add_2
identityx
mulMulmul_gru_cell_6_betamul_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_6_betamul_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
°

Ù
*__inference_gru_cell_6_layer_call_fn_51356

inputs
states_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45938o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
B
þ
while_body_49622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_6_readvariableop_resource_0:	¬D
1while_gru_cell_6_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_6_readvariableop_resource:	¬B
/while_gru_cell_6_matmul_readvariableop_resource:	¬D
1while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_6/MatMul/ReadVariableOp¢(while/gru_cell_6/MatMul_1/ReadVariableOp¢while/gru_cell_6/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_6/ReadVariableOpReadVariableOp*while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_6/unstackUnpack'while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_6/BiasAddBiasAdd!while/gru_cell_6/MatMul:product:0!while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_6/splitSplit)while/gru_cell_6/split/split_dim:output:0!while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_6/MatMul_1MatMulwhile_placeholder_20while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_6/BiasAdd_1BiasAdd#while/gru_cell_6/MatMul_1:product:0!while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_6/split_1SplitV#while/gru_cell_6/BiasAdd_1:output:0while/gru_cell_6/Const:output:0+while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_6/addAddV2while/gru_cell_6/split:output:0!while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_6/SigmoidSigmoidwhile/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_1AddV2while/gru_cell_6/split:output:1!while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_1Sigmoidwhile/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mulMulwhile/gru_cell_6/Sigmoid_1:y:0!while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_2AddV2while/gru_cell_6/split:output:2while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/mul_1Mulwhile/gru_cell_6/beta:output:0while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_2Sigmoidwhile/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_2Mulwhile/gru_cell_6/add_2:z:0while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/IdentityIdentitywhile/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_6/IdentityN	IdentityNwhile/gru_cell_6/mul_2:z:0while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49672*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_3Mulwhile/gru_cell_6/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/subSubwhile/gru_cell_6/sub/x:output:0while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_4Mulwhile/gru_cell_6/sub:z:0#while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_3AddV2while/gru_cell_6/mul_3:z:0while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_6/MatMul/ReadVariableOp)^while/gru_cell_6/MatMul_1/ReadVariableOp ^while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_6_matmul_1_readvariableop_resource3while_gru_cell_6_matmul_1_readvariableop_resource_0"d
/while_gru_cell_6_matmul_readvariableop_resource1while_gru_cell_6_matmul_readvariableop_resource_0"V
(while_gru_cell_6_readvariableop_resource*while_gru_cell_6_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_6/MatMul/ReadVariableOp&while/gru_cell_6/MatMul/ReadVariableOp2T
(while/gru_cell_6/MatMul_1/ReadVariableOp(while/gru_cell_6/MatMul_1/ReadVariableOp2B
while/gru_cell_6/ReadVariableOpwhile/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 

w
"__inference_internal_grad_fn_52960
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
Û
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51402

inputs
states_0*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51388*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdU
mul_3MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
Ø

"__inference_internal_grad_fn_52402
result_grads_0
result_grads_1
mul_gru_cell_6_beta
mul_gru_cell_6_add_2
identityx
mulMulmul_gru_cell_6_betamul_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_6_betamul_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
J
²	
gru_6_while_body_48716(
$gru_6_while_gru_6_while_loop_counter.
*gru_6_while_gru_6_while_maximum_iterations
gru_6_while_placeholder
gru_6_while_placeholder_1
gru_6_while_placeholder_2'
#gru_6_while_gru_6_strided_slice_1_0c
_gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_0C
0gru_6_while_gru_cell_6_readvariableop_resource_0:	¬J
7gru_6_while_gru_cell_6_matmul_readvariableop_resource_0:	¬L
9gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
gru_6_while_identity
gru_6_while_identity_1
gru_6_while_identity_2
gru_6_while_identity_3
gru_6_while_identity_4%
!gru_6_while_gru_6_strided_slice_1a
]gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensorA
.gru_6_while_gru_cell_6_readvariableop_resource:	¬H
5gru_6_while_gru_cell_6_matmul_readvariableop_resource:	¬J
7gru_6_while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢,gru_6/while/gru_cell_6/MatMul/ReadVariableOp¢.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp¢%gru_6/while/gru_cell_6/ReadVariableOp
=gru_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/gru_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_0gru_6_while_placeholderFgru_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%gru_6/while/gru_cell_6/ReadVariableOpReadVariableOp0gru_6_while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_6/while/gru_cell_6/unstackUnpack-gru_6/while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_6/while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp7gru_6_while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0È
gru_6/while/gru_cell_6/MatMulMatMul6gru_6/while/TensorArrayV2Read/TensorListGetItem:item:04gru_6/while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_6/while/gru_cell_6/BiasAddBiasAdd'gru_6/while/gru_cell_6/MatMul:product:0'gru_6/while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_6/while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_6/while/gru_cell_6/splitSplit/gru_6/while/gru_cell_6/split/split_dim:output:0'gru_6/while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp9gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_6/while/gru_cell_6/MatMul_1MatMulgru_6_while_placeholder_26gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_6/while/gru_cell_6/BiasAdd_1BiasAdd)gru_6/while/gru_cell_6/MatMul_1:product:0'gru_6/while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_6/while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_6/while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_6/while/gru_cell_6/split_1SplitV)gru_6/while/gru_cell_6/BiasAdd_1:output:0%gru_6/while/gru_cell_6/Const:output:01gru_6/while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_6/while/gru_cell_6/addAddV2%gru_6/while/gru_cell_6/split:output:0'gru_6/while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_6/while/gru_cell_6/SigmoidSigmoidgru_6/while/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_6/while/gru_cell_6/add_1AddV2%gru_6/while/gru_cell_6/split:output:1'gru_6/while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_6/while/gru_cell_6/Sigmoid_1Sigmoid gru_6/while/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_6/while/gru_cell_6/mulMul$gru_6/while/gru_cell_6/Sigmoid_1:y:0'gru_6/while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/add_2AddV2%gru_6/while/gru_cell_6/split:output:2gru_6/while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_6/while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/while/gru_cell_6/mul_1Mul$gru_6/while/gru_cell_6/beta:output:0 gru_6/while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_6/while/gru_cell_6/Sigmoid_2Sigmoid gru_6/while/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/mul_2Mul gru_6/while/gru_cell_6/add_2:z:0$gru_6/while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/IdentityIdentity gru_6/while/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdç
 gru_6/while/gru_cell_6/IdentityN	IdentityN gru_6/while/gru_cell_6/mul_2:z:0 gru_6/while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48766*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/mul_3Mul"gru_6/while/gru_cell_6/Sigmoid:y:0gru_6_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_6/while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/while/gru_cell_6/subSub%gru_6/while/gru_cell_6/sub/x:output:0"gru_6/while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_6/while/gru_cell_6/mul_4Mulgru_6/while/gru_cell_6/sub:z:0)gru_6/while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/add_3AddV2 gru_6/while/gru_cell_6/mul_3:z:0 gru_6/while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_6_while_placeholder_1gru_6_while_placeholder gru_6/while/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_6/while/addAddV2gru_6_while_placeholdergru_6/while/add/y:output:0*
T0*
_output_shapes
: U
gru_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_6/while/add_1AddV2$gru_6_while_gru_6_while_loop_countergru_6/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_6/while/IdentityIdentitygru_6/while/add_1:z:0^gru_6/while/NoOp*
T0*
_output_shapes
: 
gru_6/while/Identity_1Identity*gru_6_while_gru_6_while_maximum_iterations^gru_6/while/NoOp*
T0*
_output_shapes
: k
gru_6/while/Identity_2Identitygru_6/while/add:z:0^gru_6/while/NoOp*
T0*
_output_shapes
: «
gru_6/while/Identity_3Identity@gru_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_6/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_6/while/Identity_4Identity gru_6/while/gru_cell_6/add_3:z:0^gru_6/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_6/while/NoOpNoOp-^gru_6/while/gru_cell_6/MatMul/ReadVariableOp/^gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp&^gru_6/while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_6_while_gru_6_strided_slice_1#gru_6_while_gru_6_strided_slice_1_0"t
7gru_6_while_gru_cell_6_matmul_1_readvariableop_resource9gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0"p
5gru_6_while_gru_cell_6_matmul_readvariableop_resource7gru_6_while_gru_cell_6_matmul_readvariableop_resource_0"b
.gru_6_while_gru_cell_6_readvariableop_resource0gru_6_while_gru_cell_6_readvariableop_resource_0"5
gru_6_while_identitygru_6/while/Identity:output:0"9
gru_6_while_identity_1gru_6/while/Identity_1:output:0"9
gru_6_while_identity_2gru_6/while/Identity_2:output:0"9
gru_6_while_identity_3gru_6/while/Identity_3:output:0"9
gru_6_while_identity_4gru_6/while/Identity_4:output:0"À
]gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_6/while/gru_cell_6/MatMul/ReadVariableOp,gru_6/while/gru_cell_6/MatMul/ReadVariableOp2`
.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp2N
%gru_6/while/gru_cell_6/ReadVariableOp%gru_6/while/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Ì
³
"__inference_internal_grad_fn_51826
result_grads_0
result_grads_1*
&mul_sequential_2_gru_8_gru_cell_8_beta+
'mul_sequential_2_gru_8_gru_cell_8_add_2
identity
mulMul&mul_sequential_2_gru_8_gru_cell_8_beta'mul_sequential_2_gru_8_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul&mul_sequential_2_gru_8_gru_cell_8_beta'mul_sequential_2_gru_8_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

w
"__inference_internal_grad_fn_52906
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
æ[
¸
#sequential_2_gru_7_while_body_45446B
>sequential_2_gru_7_while_sequential_2_gru_7_while_loop_counterH
Dsequential_2_gru_7_while_sequential_2_gru_7_while_maximum_iterations(
$sequential_2_gru_7_while_placeholder*
&sequential_2_gru_7_while_placeholder_1*
&sequential_2_gru_7_while_placeholder_2A
=sequential_2_gru_7_while_sequential_2_gru_7_strided_slice_1_0}
ysequential_2_gru_7_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_7_tensorarrayunstack_tensorlistfromtensor_0P
=sequential_2_gru_7_while_gru_cell_7_readvariableop_resource_0:	¬W
Dsequential_2_gru_7_while_gru_cell_7_matmul_readvariableop_resource_0:	d¬Y
Fsequential_2_gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬%
!sequential_2_gru_7_while_identity'
#sequential_2_gru_7_while_identity_1'
#sequential_2_gru_7_while_identity_2'
#sequential_2_gru_7_while_identity_3'
#sequential_2_gru_7_while_identity_4?
;sequential_2_gru_7_while_sequential_2_gru_7_strided_slice_1{
wsequential_2_gru_7_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_7_tensorarrayunstack_tensorlistfromtensorN
;sequential_2_gru_7_while_gru_cell_7_readvariableop_resource:	¬U
Bsequential_2_gru_7_while_gru_cell_7_matmul_readvariableop_resource:	d¬W
Dsequential_2_gru_7_while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢9sequential_2/gru_7/while/gru_cell_7/MatMul/ReadVariableOp¢;sequential_2/gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp¢2sequential_2/gru_7/while/gru_cell_7/ReadVariableOp
Jsequential_2/gru_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
<sequential_2/gru_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_2_gru_7_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_7_tensorarrayunstack_tensorlistfromtensor_0$sequential_2_gru_7_while_placeholderSsequential_2/gru_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0±
2sequential_2/gru_7/while/gru_cell_7/ReadVariableOpReadVariableOp=sequential_2_gru_7_while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0©
+sequential_2/gru_7/while/gru_cell_7/unstackUnpack:sequential_2/gru_7/while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¿
9sequential_2/gru_7/while/gru_cell_7/MatMul/ReadVariableOpReadVariableOpDsequential_2_gru_7_while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0ï
*sequential_2/gru_7/while/gru_cell_7/MatMulMatMulCsequential_2/gru_7/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_2/gru_7/while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Õ
+sequential_2/gru_7/while/gru_cell_7/BiasAddBiasAdd4sequential_2/gru_7/while/gru_cell_7/MatMul:product:04sequential_2/gru_7/while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
3sequential_2/gru_7/while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
)sequential_2/gru_7/while/gru_cell_7/splitSplit<sequential_2/gru_7/while/gru_cell_7/split/split_dim:output:04sequential_2/gru_7/while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÃ
;sequential_2/gru_7/while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOpFsequential_2_gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ö
,sequential_2/gru_7/while/gru_cell_7/MatMul_1MatMul&sequential_2_gru_7_while_placeholder_2Csequential_2/gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ù
-sequential_2/gru_7/while/gru_cell_7/BiasAdd_1BiasAdd6sequential_2/gru_7/while/gru_cell_7/MatMul_1:product:04sequential_2/gru_7/while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
)sequential_2/gru_7/while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ
5sequential_2/gru_7/while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
+sequential_2/gru_7/while/gru_cell_7/split_1SplitV6sequential_2/gru_7/while/gru_cell_7/BiasAdd_1:output:02sequential_2/gru_7/while/gru_cell_7/Const:output:0>sequential_2/gru_7/while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÌ
'sequential_2/gru_7/while/gru_cell_7/addAddV22sequential_2/gru_7/while/gru_cell_7/split:output:04sequential_2/gru_7/while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+sequential_2/gru_7/while/gru_cell_7/SigmoidSigmoid+sequential_2/gru_7/while/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÎ
)sequential_2/gru_7/while/gru_cell_7/add_1AddV22sequential_2/gru_7/while/gru_cell_7/split:output:14sequential_2/gru_7/while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_7/while/gru_cell_7/Sigmoid_1Sigmoid-sequential_2/gru_7/while/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÉ
'sequential_2/gru_7/while/gru_cell_7/mulMul1sequential_2/gru_7/while/gru_cell_7/Sigmoid_1:y:04sequential_2/gru_7/while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ
)sequential_2/gru_7/while/gru_cell_7/add_2AddV22sequential_2/gru_7/while/gru_cell_7/split:output:2+sequential_2/gru_7/while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(sequential_2/gru_7/while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
)sequential_2/gru_7/while/gru_cell_7/mul_1Mul1sequential_2/gru_7/while/gru_cell_7/beta:output:0-sequential_2/gru_7/while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_7/while/gru_cell_7/Sigmoid_2Sigmoid-sequential_2/gru_7/while/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
)sequential_2/gru_7/while/gru_cell_7/mul_2Mul-sequential_2/gru_7/while/gru_cell_7/add_2:z:01sequential_2/gru_7/while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_2/gru_7/while/gru_cell_7/IdentityIdentity-sequential_2/gru_7/while/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_2/gru_7/while/gru_cell_7/IdentityN	IdentityN-sequential_2/gru_7/while/gru_cell_7/mul_2:z:0-sequential_2/gru_7/while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45496*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd»
)sequential_2/gru_7/while/gru_cell_7/mul_3Mul/sequential_2/gru_7/while/gru_cell_7/Sigmoid:y:0&sequential_2_gru_7_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
)sequential_2/gru_7/while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Å
'sequential_2/gru_7/while/gru_cell_7/subSub2sequential_2/gru_7/while/gru_cell_7/sub/x:output:0/sequential_2/gru_7/while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
)sequential_2/gru_7/while/gru_cell_7/mul_4Mul+sequential_2/gru_7/while/gru_cell_7/sub:z:06sequential_2/gru_7/while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ
)sequential_2/gru_7/while/gru_cell_7/add_3AddV2-sequential_2/gru_7/while/gru_cell_7/mul_3:z:0-sequential_2/gru_7/while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
=sequential_2/gru_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_2_gru_7_while_placeholder_1$sequential_2_gru_7_while_placeholder-sequential_2/gru_7/while/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_2/gru_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_2/gru_7/while/addAddV2$sequential_2_gru_7_while_placeholder'sequential_2/gru_7/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_2/gru_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_2/gru_7/while/add_1AddV2>sequential_2_gru_7_while_sequential_2_gru_7_while_loop_counter)sequential_2/gru_7/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_2/gru_7/while/IdentityIdentity"sequential_2/gru_7/while/add_1:z:0^sequential_2/gru_7/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_2/gru_7/while/Identity_1IdentityDsequential_2_gru_7_while_sequential_2_gru_7_while_maximum_iterations^sequential_2/gru_7/while/NoOp*
T0*
_output_shapes
: 
#sequential_2/gru_7/while/Identity_2Identity sequential_2/gru_7/while/add:z:0^sequential_2/gru_7/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_2/gru_7/while/Identity_3IdentityMsequential_2/gru_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_2/gru_7/while/NoOp*
T0*
_output_shapes
: :éèÒ°
#sequential_2/gru_7/while/Identity_4Identity-sequential_2/gru_7/while/gru_cell_7/add_3:z:0^sequential_2/gru_7/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_2/gru_7/while/NoOpNoOp:^sequential_2/gru_7/while/gru_cell_7/MatMul/ReadVariableOp<^sequential_2/gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp3^sequential_2/gru_7/while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Dsequential_2_gru_7_while_gru_cell_7_matmul_1_readvariableop_resourceFsequential_2_gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0"
Bsequential_2_gru_7_while_gru_cell_7_matmul_readvariableop_resourceDsequential_2_gru_7_while_gru_cell_7_matmul_readvariableop_resource_0"|
;sequential_2_gru_7_while_gru_cell_7_readvariableop_resource=sequential_2_gru_7_while_gru_cell_7_readvariableop_resource_0"O
!sequential_2_gru_7_while_identity*sequential_2/gru_7/while/Identity:output:0"S
#sequential_2_gru_7_while_identity_1,sequential_2/gru_7/while/Identity_1:output:0"S
#sequential_2_gru_7_while_identity_2,sequential_2/gru_7/while/Identity_2:output:0"S
#sequential_2_gru_7_while_identity_3,sequential_2/gru_7/while/Identity_3:output:0"S
#sequential_2_gru_7_while_identity_4,sequential_2/gru_7/while/Identity_4:output:0"|
;sequential_2_gru_7_while_sequential_2_gru_7_strided_slice_1=sequential_2_gru_7_while_sequential_2_gru_7_strided_slice_1_0"ô
wsequential_2_gru_7_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_7_tensorarrayunstack_tensorlistfromtensorysequential_2_gru_7_while_tensorarrayv2read_tensorlistgetitem_sequential_2_gru_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2v
9sequential_2/gru_7/while/gru_cell_7/MatMul/ReadVariableOp9sequential_2/gru_7/while/gru_cell_7/MatMul/ReadVariableOp2z
;sequential_2/gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp;sequential_2/gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp2h
2sequential_2/gru_7/while/gru_cell_7/ReadVariableOp2sequential_2/gru_7/while/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
§
¸
%__inference_gru_6_layer_call_fn_49195
inputs_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_46054|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
R

@__inference_gru_6_layer_call_and_return_conditional_losses_49551
inputs_05
"gru_cell_6_readvariableop_resource:	¬<
)gru_cell_6_matmul_readvariableop_resource:	¬>
+gru_cell_6_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_6/MatMul/ReadVariableOp¢"gru_cell_6/MatMul_1/ReadVariableOp¢gru_cell_6/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_6/ReadVariableOpReadVariableOp"gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_6/unstackUnpack!gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_6/MatMul/ReadVariableOpReadVariableOp)gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_6/MatMulMatMulstrided_slice_2:output:0(gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAddBiasAddgru_cell_6/MatMul:product:0gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_6/splitSplit#gru_cell_6/split/split_dim:output:0gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_6/MatMul_1MatMulzeros:output:0*gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAdd_1BiasAddgru_cell_6/MatMul_1:product:0gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_6/split_1SplitVgru_cell_6/BiasAdd_1:output:0gru_cell_6/Const:output:0%gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_6/addAddV2gru_cell_6/split:output:0gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_6/SigmoidSigmoidgru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_6/add_1AddV2gru_cell_6/split:output:1gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_1Sigmoidgru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_6/mulMulgru_cell_6/Sigmoid_1:y:0gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_6/add_2AddV2gru_cell_6/split:output:2gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_6/mul_1Mulgru_cell_6/beta:output:0gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_2Sigmoidgru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_6/mul_2Mulgru_cell_6/add_2:z:0gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/IdentityIdentitygru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_6/IdentityN	IdentityNgru_cell_6/mul_2:z:0gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49439*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_6/mul_3Mulgru_cell_6/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_6/subSubgru_cell_6/sub/x:output:0gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_6/mul_4Mulgru_cell_6/sub:z:0gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_6/add_3AddV2gru_cell_6/mul_3:z:0gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_6_readvariableop_resource)gru_cell_6_matmul_readvariableop_resource+gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_49455*
condR
while_cond_49454*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_6/MatMul/ReadVariableOp#^gru_cell_6/MatMul_1/ReadVariableOp^gru_cell_6/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_6/MatMul/ReadVariableOp gru_cell_6/MatMul/ReadVariableOp2H
"gru_cell_6/MatMul_1/ReadVariableOp"gru_cell_6/MatMul_1/ReadVariableOp26
gru_cell_6/ReadVariableOpgru_cell_6/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
÷

gru_8_while_cond_48542(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2*
&gru_8_while_less_gru_8_strided_slice_1?
;gru_8_while_gru_8_while_cond_48542___redundant_placeholder0?
;gru_8_while_gru_8_while_cond_48542___redundant_placeholder1?
;gru_8_while_gru_8_while_cond_48542___redundant_placeholder2?
;gru_8_while_gru_8_while_cond_48542___redundant_placeholder3
gru_8_while_identity
z
gru_8/while/LessLessgru_8_while_placeholder&gru_8_while_less_gru_8_strided_slice_1*
T0*
_output_shapes
: W
gru_8/while/IdentityIdentitygru_8/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_8_while_identitygru_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
!
Ù
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46290

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-46276*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdS
mul_3MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates

ù	
G__inference_sequential_2_layer_call_and_return_conditional_losses_49144

inputs;
(gru_6_gru_cell_6_readvariableop_resource:	¬B
/gru_6_gru_cell_6_matmul_readvariableop_resource:	¬D
1gru_6_gru_cell_6_matmul_1_readvariableop_resource:	d¬;
(gru_7_gru_cell_7_readvariableop_resource:	¬B
/gru_7_gru_cell_7_matmul_readvariableop_resource:	d¬D
1gru_7_gru_cell_7_matmul_1_readvariableop_resource:	d¬;
(gru_8_gru_cell_8_readvariableop_resource:	¬B
/gru_8_gru_cell_8_matmul_readvariableop_resource:	d¬D
1gru_8_gru_cell_8_matmul_1_readvariableop_resource:	d¬8
&dense_2_matmul_readvariableop_resource:d5
'dense_2_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢&gru_6/gru_cell_6/MatMul/ReadVariableOp¢(gru_6/gru_cell_6/MatMul_1/ReadVariableOp¢gru_6/gru_cell_6/ReadVariableOp¢gru_6/while¢&gru_7/gru_cell_7/MatMul/ReadVariableOp¢(gru_7/gru_cell_7/MatMul_1/ReadVariableOp¢gru_7/gru_cell_7/ReadVariableOp¢gru_7/while¢&gru_8/gru_cell_8/MatMul/ReadVariableOp¢(gru_8/gru_cell_8/MatMul_1/ReadVariableOp¢gru_8/gru_cell_8/ReadVariableOp¢gru_8/whileA
gru_6/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_6/strided_sliceStridedSlicegru_6/Shape:output:0"gru_6/strided_slice/stack:output:0$gru_6/strided_slice/stack_1:output:0$gru_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_6/zeros/packedPackgru_6/strided_slice:output:0gru_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_6/zerosFillgru_6/zeros/packed:output:0gru_6/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_6/transpose	Transposeinputsgru_6/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿP
gru_6/Shape_1Shapegru_6/transpose:y:0*
T0*
_output_shapes
:e
gru_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_6/strided_slice_1StridedSlicegru_6/Shape_1:output:0$gru_6/strided_slice_1/stack:output:0&gru_6/strided_slice_1/stack_1:output:0&gru_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_6/TensorArrayV2TensorListReserve*gru_6/TensorArrayV2/element_shape:output:0gru_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_6/transpose:y:0Dgru_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_6/strided_slice_2StridedSlicegru_6/transpose:y:0$gru_6/strided_slice_2/stack:output:0&gru_6/strided_slice_2/stack_1:output:0&gru_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_6/gru_cell_6/ReadVariableOpReadVariableOp(gru_6_gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_6/gru_cell_6/unstackUnpack'gru_6/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_6/gru_cell_6/MatMul/ReadVariableOpReadVariableOp/gru_6_gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0¤
gru_6/gru_cell_6/MatMulMatMulgru_6/strided_slice_2:output:0.gru_6/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_6/gru_cell_6/BiasAddBiasAdd!gru_6/gru_cell_6/MatMul:product:0!gru_6/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_6/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_6/gru_cell_6/splitSplit)gru_6/gru_cell_6/split/split_dim:output:0!gru_6/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_6/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp1gru_6_gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_6/gru_cell_6/MatMul_1MatMulgru_6/zeros:output:00gru_6/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_6/gru_cell_6/BiasAdd_1BiasAdd#gru_6/gru_cell_6/MatMul_1:product:0!gru_6/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_6/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_6/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_6/gru_cell_6/split_1SplitV#gru_6/gru_cell_6/BiasAdd_1:output:0gru_6/gru_cell_6/Const:output:0+gru_6/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_6/gru_cell_6/addAddV2gru_6/gru_cell_6/split:output:0!gru_6/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_6/gru_cell_6/SigmoidSigmoidgru_6/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/add_1AddV2gru_6/gru_cell_6/split:output:1!gru_6/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_6/gru_cell_6/Sigmoid_1Sigmoidgru_6/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mulMulgru_6/gru_cell_6/Sigmoid_1:y:0!gru_6/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/add_2AddV2gru_6/gru_cell_6/split:output:2gru_6/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_6/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/gru_cell_6/mul_1Mulgru_6/gru_cell_6/beta:output:0gru_6/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_6/gru_cell_6/Sigmoid_2Sigmoidgru_6/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mul_2Mulgru_6/gru_cell_6/add_2:z:0gru_6/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_6/gru_cell_6/IdentityIdentitygru_6/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
gru_6/gru_cell_6/IdentityN	IdentityNgru_6/gru_cell_6/mul_2:z:0gru_6/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48700*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mul_3Mulgru_6/gru_cell_6/Sigmoid:y:0gru_6/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_6/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/gru_cell_6/subSubgru_6/gru_cell_6/sub/x:output:0gru_6/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mul_4Mulgru_6/gru_cell_6/sub:z:0#gru_6/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/add_3AddV2gru_6/gru_cell_6/mul_3:z:0gru_6/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_6/TensorArrayV2_1TensorListReserve,gru_6/TensorArrayV2_1/element_shape:output:0gru_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_6/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_6/whileWhile!gru_6/while/loop_counter:output:0'gru_6/while/maximum_iterations:output:0gru_6/time:output:0gru_6/TensorArrayV2_1:handle:0gru_6/zeros:output:0gru_6/strided_slice_1:output:0=gru_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_6_gru_cell_6_readvariableop_resource/gru_6_gru_cell_6_matmul_readvariableop_resource1gru_6_gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *"
bodyR
gru_6_while_body_48716*"
condR
gru_6_while_cond_48715*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_6/TensorArrayV2Stack/TensorListStackTensorListStackgru_6/while:output:3?gru_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_6/strided_slice_3StridedSlice1gru_6/TensorArrayV2Stack/TensorListStack:tensor:0$gru_6/strided_slice_3/stack:output:0&gru_6/strided_slice_3/stack_1:output:0&gru_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_6/transpose_1	Transpose1gru_6/TensorArrayV2Stack/TensorListStack:tensor:0gru_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_7/ShapeShapegru_6/transpose_1:y:0*
T0*
_output_shapes
:c
gru_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_7/strided_sliceStridedSlicegru_7/Shape:output:0"gru_7/strided_slice/stack:output:0$gru_7/strided_slice/stack_1:output:0$gru_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_7/zeros/packedPackgru_7/strided_slice:output:0gru_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_7/zerosFillgru_7/zeros/packed:output:0gru_7/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_7/transpose	Transposegru_6/transpose_1:y:0gru_7/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_7/Shape_1Shapegru_7/transpose:y:0*
T0*
_output_shapes
:e
gru_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_7/strided_slice_1StridedSlicegru_7/Shape_1:output:0$gru_7/strided_slice_1/stack:output:0&gru_7/strided_slice_1/stack_1:output:0&gru_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_7/TensorArrayV2TensorListReserve*gru_7/TensorArrayV2/element_shape:output:0gru_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_7/transpose:y:0Dgru_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_7/strided_slice_2StridedSlicegru_7/transpose:y:0$gru_7/strided_slice_2/stack:output:0&gru_7/strided_slice_2/stack_1:output:0&gru_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
gru_7/gru_cell_7/ReadVariableOpReadVariableOp(gru_7_gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_7/gru_cell_7/unstackUnpack'gru_7/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_7/gru_cell_7/MatMul/ReadVariableOpReadVariableOp/gru_7_gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¤
gru_7/gru_cell_7/MatMulMatMulgru_7/strided_slice_2:output:0.gru_7/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_7/gru_cell_7/BiasAddBiasAdd!gru_7/gru_cell_7/MatMul:product:0!gru_7/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_7/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_7/gru_cell_7/splitSplit)gru_7/gru_cell_7/split/split_dim:output:0!gru_7/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_7/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp1gru_7_gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_7/gru_cell_7/MatMul_1MatMulgru_7/zeros:output:00gru_7/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_7/gru_cell_7/BiasAdd_1BiasAdd#gru_7/gru_cell_7/MatMul_1:product:0!gru_7/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_7/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_7/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_7/gru_cell_7/split_1SplitV#gru_7/gru_cell_7/BiasAdd_1:output:0gru_7/gru_cell_7/Const:output:0+gru_7/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_7/gru_cell_7/addAddV2gru_7/gru_cell_7/split:output:0!gru_7/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_7/gru_cell_7/SigmoidSigmoidgru_7/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/add_1AddV2gru_7/gru_cell_7/split:output:1!gru_7/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_7/gru_cell_7/Sigmoid_1Sigmoidgru_7/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mulMulgru_7/gru_cell_7/Sigmoid_1:y:0!gru_7/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/add_2AddV2gru_7/gru_cell_7/split:output:2gru_7/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_7/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/gru_cell_7/mul_1Mulgru_7/gru_cell_7/beta:output:0gru_7/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_7/gru_cell_7/Sigmoid_2Sigmoidgru_7/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mul_2Mulgru_7/gru_cell_7/add_2:z:0gru_7/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_7/gru_cell_7/IdentityIdentitygru_7/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
gru_7/gru_cell_7/IdentityN	IdentityNgru_7/gru_cell_7/mul_2:z:0gru_7/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48863*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mul_3Mulgru_7/gru_cell_7/Sigmoid:y:0gru_7/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_7/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/gru_cell_7/subSubgru_7/gru_cell_7/sub/x:output:0gru_7/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mul_4Mulgru_7/gru_cell_7/sub:z:0#gru_7/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/add_3AddV2gru_7/gru_cell_7/mul_3:z:0gru_7/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_7/TensorArrayV2_1TensorListReserve,gru_7/TensorArrayV2_1/element_shape:output:0gru_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_7/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_7/whileWhile!gru_7/while/loop_counter:output:0'gru_7/while/maximum_iterations:output:0gru_7/time:output:0gru_7/TensorArrayV2_1:handle:0gru_7/zeros:output:0gru_7/strided_slice_1:output:0=gru_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_7_gru_cell_7_readvariableop_resource/gru_7_gru_cell_7_matmul_readvariableop_resource1gru_7_gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *"
bodyR
gru_7_while_body_48879*"
condR
gru_7_while_cond_48878*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_7/TensorArrayV2Stack/TensorListStackTensorListStackgru_7/while:output:3?gru_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_7/strided_slice_3StridedSlice1gru_7/TensorArrayV2Stack/TensorListStack:tensor:0$gru_7/strided_slice_3/stack:output:0&gru_7/strided_slice_3/stack_1:output:0&gru_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_7/transpose_1	Transpose1gru_7/TensorArrayV2Stack/TensorListStack:tensor:0gru_7/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_7/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_8/ShapeShapegru_7/transpose_1:y:0*
T0*
_output_shapes
:c
gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_8/strided_sliceStridedSlicegru_8/Shape:output:0"gru_8/strided_slice/stack:output:0$gru_8/strided_slice/stack_1:output:0$gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_8/zeros/packedPackgru_8/strided_slice:output:0gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_8/zerosFillgru_8/zeros/packed:output:0gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_8/transpose	Transposegru_7/transpose_1:y:0gru_8/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_8/Shape_1Shapegru_8/transpose:y:0*
T0*
_output_shapes
:e
gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_8/strided_slice_1StridedSlicegru_8/Shape_1:output:0$gru_8/strided_slice_1/stack:output:0&gru_8/strided_slice_1/stack_1:output:0&gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_8/TensorArrayV2TensorListReserve*gru_8/TensorArrayV2/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_8/transpose:y:0Dgru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_8/strided_slice_2StridedSlicegru_8/transpose:y:0$gru_8/strided_slice_2/stack:output:0&gru_8/strided_slice_2/stack_1:output:0&gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
gru_8/gru_cell_8/ReadVariableOpReadVariableOp(gru_8_gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_8/gru_cell_8/unstackUnpack'gru_8/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_8/gru_cell_8/MatMul/ReadVariableOpReadVariableOp/gru_8_gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¤
gru_8/gru_cell_8/MatMulMatMulgru_8/strided_slice_2:output:0.gru_8/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_8/gru_cell_8/BiasAddBiasAdd!gru_8/gru_cell_8/MatMul:product:0!gru_8/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_8/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_8/gru_cell_8/splitSplit)gru_8/gru_cell_8/split/split_dim:output:0!gru_8/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_8/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp1gru_8_gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_8/gru_cell_8/MatMul_1MatMulgru_8/zeros:output:00gru_8/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_8/gru_cell_8/BiasAdd_1BiasAdd#gru_8/gru_cell_8/MatMul_1:product:0!gru_8/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_8/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_8/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_8/gru_cell_8/split_1SplitV#gru_8/gru_cell_8/BiasAdd_1:output:0gru_8/gru_cell_8/Const:output:0+gru_8/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_8/gru_cell_8/addAddV2gru_8/gru_cell_8/split:output:0!gru_8/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_8/gru_cell_8/SigmoidSigmoidgru_8/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/add_1AddV2gru_8/gru_cell_8/split:output:1!gru_8/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_8/gru_cell_8/Sigmoid_1Sigmoidgru_8/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mulMulgru_8/gru_cell_8/Sigmoid_1:y:0!gru_8/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/add_2AddV2gru_8/gru_cell_8/split:output:2gru_8/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_8/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/gru_cell_8/mul_1Mulgru_8/gru_cell_8/beta:output:0gru_8/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_8/gru_cell_8/Sigmoid_2Sigmoidgru_8/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mul_2Mulgru_8/gru_cell_8/add_2:z:0gru_8/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_8/gru_cell_8/IdentityIdentitygru_8/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
gru_8/gru_cell_8/IdentityN	IdentityNgru_8/gru_cell_8/mul_2:z:0gru_8/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49026*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mul_3Mulgru_8/gru_cell_8/Sigmoid:y:0gru_8/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_8/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/gru_cell_8/subSubgru_8/gru_cell_8/sub/x:output:0gru_8/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mul_4Mulgru_8/gru_cell_8/sub:z:0#gru_8/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/add_3AddV2gru_8/gru_cell_8/mul_3:z:0gru_8/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_8/TensorArrayV2_1TensorListReserve,gru_8/TensorArrayV2_1/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_8/whileWhile!gru_8/while/loop_counter:output:0'gru_8/while/maximum_iterations:output:0gru_8/time:output:0gru_8/TensorArrayV2_1:handle:0gru_8/zeros:output:0gru_8/strided_slice_1:output:0=gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_8_gru_cell_8_readvariableop_resource/gru_8_gru_cell_8_matmul_readvariableop_resource1gru_8_gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *"
bodyR
gru_8_while_body_49042*"
condR
gru_8_while_cond_49041*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_8/TensorArrayV2Stack/TensorListStackTensorListStackgru_8/while:output:3?gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_8/strided_slice_3StridedSlice1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0$gru_8/strided_slice_3/stack:output:0&gru_8/strided_slice_3/stack_1:output:0&gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_8/transpose_1	Transpose1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0gru_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_2/MatMulMatMulgru_8/strided_slice_3:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^gru_6/gru_cell_6/MatMul/ReadVariableOp)^gru_6/gru_cell_6/MatMul_1/ReadVariableOp ^gru_6/gru_cell_6/ReadVariableOp^gru_6/while'^gru_7/gru_cell_7/MatMul/ReadVariableOp)^gru_7/gru_cell_7/MatMul_1/ReadVariableOp ^gru_7/gru_cell_7/ReadVariableOp^gru_7/while'^gru_8/gru_cell_8/MatMul/ReadVariableOp)^gru_8/gru_cell_8/MatMul_1/ReadVariableOp ^gru_8/gru_cell_8/ReadVariableOp^gru_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
&gru_6/gru_cell_6/MatMul/ReadVariableOp&gru_6/gru_cell_6/MatMul/ReadVariableOp2T
(gru_6/gru_cell_6/MatMul_1/ReadVariableOp(gru_6/gru_cell_6/MatMul_1/ReadVariableOp2B
gru_6/gru_cell_6/ReadVariableOpgru_6/gru_cell_6/ReadVariableOp2
gru_6/whilegru_6/while2P
&gru_7/gru_cell_7/MatMul/ReadVariableOp&gru_7/gru_cell_7/MatMul/ReadVariableOp2T
(gru_7/gru_cell_7/MatMul_1/ReadVariableOp(gru_7/gru_cell_7/MatMul_1/ReadVariableOp2B
gru_7/gru_cell_7/ReadVariableOpgru_7/gru_cell_7/ReadVariableOp2
gru_7/whilegru_7/while2P
&gru_8/gru_cell_8/MatMul/ReadVariableOp&gru_8/gru_cell_8/MatMul/ReadVariableOp2T
(gru_8/gru_cell_8/MatMul_1/ReadVariableOp(gru_8/gru_cell_8/MatMul_1/ReadVariableOp2B
gru_8/gru_cell_8/ReadVariableOpgru_8/gru_cell_8/ReadVariableOp2
gru_8/whilegru_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
×

#sequential_2_gru_7_while_cond_45445B
>sequential_2_gru_7_while_sequential_2_gru_7_while_loop_counterH
Dsequential_2_gru_7_while_sequential_2_gru_7_while_maximum_iterations(
$sequential_2_gru_7_while_placeholder*
&sequential_2_gru_7_while_placeholder_1*
&sequential_2_gru_7_while_placeholder_2D
@sequential_2_gru_7_while_less_sequential_2_gru_7_strided_slice_1Y
Usequential_2_gru_7_while_sequential_2_gru_7_while_cond_45445___redundant_placeholder0Y
Usequential_2_gru_7_while_sequential_2_gru_7_while_cond_45445___redundant_placeholder1Y
Usequential_2_gru_7_while_sequential_2_gru_7_while_cond_45445___redundant_placeholder2Y
Usequential_2_gru_7_while_sequential_2_gru_7_while_cond_45445___redundant_placeholder3%
!sequential_2_gru_7_while_identity
®
sequential_2/gru_7/while/LessLess$sequential_2_gru_7_while_placeholder@sequential_2_gru_7_while_less_sequential_2_gru_7_strided_slice_1*
T0*
_output_shapes
: q
!sequential_2/gru_7/while/IdentityIdentity!sequential_2/gru_7/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_2_gru_7_while_identity*sequential_2/gru_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ÆQ

@__inference_gru_7_layer_call_and_return_conditional_losses_50597

inputs5
"gru_cell_7_readvariableop_resource:	¬<
)gru_cell_7_matmul_readvariableop_resource:	d¬>
+gru_cell_7_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_7/MatMul/ReadVariableOp¢"gru_cell_7/MatMul_1/ReadVariableOp¢gru_cell_7/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_7/ReadVariableOpReadVariableOp"gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_7/unstackUnpack!gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_7/MatMul/ReadVariableOpReadVariableOp)gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMulMatMulstrided_slice_2:output:0(gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAddBiasAddgru_cell_7/MatMul:product:0gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_7/splitSplit#gru_cell_7/split/split_dim:output:0gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMul_1MatMulzeros:output:0*gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAdd_1BiasAddgru_cell_7/MatMul_1:product:0gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_7/split_1SplitVgru_cell_7/BiasAdd_1:output:0gru_cell_7/Const:output:0%gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_7/addAddV2gru_cell_7/split:output:0gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_7/SigmoidSigmoidgru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_7/add_1AddV2gru_cell_7/split:output:1gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_1Sigmoidgru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_7/mulMulgru_cell_7/Sigmoid_1:y:0gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_7/add_2AddV2gru_cell_7/split:output:2gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_7/mul_1Mulgru_cell_7/beta:output:0gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_2Sigmoidgru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_7/mul_2Mulgru_cell_7/add_2:z:0gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/IdentityIdentitygru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_7/IdentityN	IdentityNgru_cell_7/mul_2:z:0gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50485*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_7/mul_3Mulgru_cell_7/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_7/subSubgru_cell_7/sub/x:output:0gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_7/mul_4Mulgru_cell_7/sub:z:0gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_7/add_3AddV2gru_cell_7/mul_3:z:0gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_7_readvariableop_resource)gru_cell_7_matmul_readvariableop_resource+gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_50501*
condR
while_cond_50500*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_7/MatMul/ReadVariableOp#^gru_cell_7/MatMul_1/ReadVariableOp^gru_cell_7/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_7/MatMul/ReadVariableOp gru_cell_7/MatMul/ReadVariableOp2H
"gru_cell_7/MatMul_1/ReadVariableOp"gru_cell_7/MatMul_1/ReadVariableOp26
gru_cell_7/ReadVariableOpgru_cell_7/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
§
¸
%__inference_gru_7_layer_call_fn_49907
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_46406|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
÷

gru_6_while_cond_48216(
$gru_6_while_gru_6_while_loop_counter.
*gru_6_while_gru_6_while_maximum_iterations
gru_6_while_placeholder
gru_6_while_placeholder_1
gru_6_while_placeholder_2*
&gru_6_while_less_gru_6_strided_slice_1?
;gru_6_while_gru_6_while_cond_48216___redundant_placeholder0?
;gru_6_while_gru_6_while_cond_48216___redundant_placeholder1?
;gru_6_while_gru_6_while_cond_48216___redundant_placeholder2?
;gru_6_while_gru_6_while_cond_48216___redundant_placeholder3
gru_6_while_identity
z
gru_6/while/LessLessgru_6_while_placeholder&gru_6_while_less_gru_6_strided_slice_1*
T0*
_output_shapes
: W
gru_6/while/IdentityIdentitygru_6/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_6_while_identitygru_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
J
²	
gru_7_while_body_48380(
$gru_7_while_gru_7_while_loop_counter.
*gru_7_while_gru_7_while_maximum_iterations
gru_7_while_placeholder
gru_7_while_placeholder_1
gru_7_while_placeholder_2'
#gru_7_while_gru_7_strided_slice_1_0c
_gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_0C
0gru_7_while_gru_cell_7_readvariableop_resource_0:	¬J
7gru_7_while_gru_cell_7_matmul_readvariableop_resource_0:	d¬L
9gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
gru_7_while_identity
gru_7_while_identity_1
gru_7_while_identity_2
gru_7_while_identity_3
gru_7_while_identity_4%
!gru_7_while_gru_7_strided_slice_1a
]gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensorA
.gru_7_while_gru_cell_7_readvariableop_resource:	¬H
5gru_7_while_gru_cell_7_matmul_readvariableop_resource:	d¬J
7gru_7_while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢,gru_7/while/gru_cell_7/MatMul/ReadVariableOp¢.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp¢%gru_7/while/gru_cell_7/ReadVariableOp
=gru_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_0gru_7_while_placeholderFgru_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
%gru_7/while/gru_cell_7/ReadVariableOpReadVariableOp0gru_7_while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_7/while/gru_cell_7/unstackUnpack-gru_7/while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_7/while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp7gru_7_while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0È
gru_7/while/gru_cell_7/MatMulMatMul6gru_7/while/TensorArrayV2Read/TensorListGetItem:item:04gru_7/while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_7/while/gru_cell_7/BiasAddBiasAdd'gru_7/while/gru_cell_7/MatMul:product:0'gru_7/while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_7/while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_7/while/gru_cell_7/splitSplit/gru_7/while/gru_cell_7/split/split_dim:output:0'gru_7/while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp9gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_7/while/gru_cell_7/MatMul_1MatMulgru_7_while_placeholder_26gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_7/while/gru_cell_7/BiasAdd_1BiasAdd)gru_7/while/gru_cell_7/MatMul_1:product:0'gru_7/while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_7/while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_7/while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_7/while/gru_cell_7/split_1SplitV)gru_7/while/gru_cell_7/BiasAdd_1:output:0%gru_7/while/gru_cell_7/Const:output:01gru_7/while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_7/while/gru_cell_7/addAddV2%gru_7/while/gru_cell_7/split:output:0'gru_7/while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_7/while/gru_cell_7/SigmoidSigmoidgru_7/while/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_7/while/gru_cell_7/add_1AddV2%gru_7/while/gru_cell_7/split:output:1'gru_7/while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_7/while/gru_cell_7/Sigmoid_1Sigmoid gru_7/while/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_7/while/gru_cell_7/mulMul$gru_7/while/gru_cell_7/Sigmoid_1:y:0'gru_7/while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/add_2AddV2%gru_7/while/gru_cell_7/split:output:2gru_7/while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_7/while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/while/gru_cell_7/mul_1Mul$gru_7/while/gru_cell_7/beta:output:0 gru_7/while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_7/while/gru_cell_7/Sigmoid_2Sigmoid gru_7/while/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/mul_2Mul gru_7/while/gru_cell_7/add_2:z:0$gru_7/while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/IdentityIdentity gru_7/while/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdç
 gru_7/while/gru_cell_7/IdentityN	IdentityN gru_7/while/gru_cell_7/mul_2:z:0 gru_7/while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48430*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/mul_3Mul"gru_7/while/gru_cell_7/Sigmoid:y:0gru_7_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_7/while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/while/gru_cell_7/subSub%gru_7/while/gru_cell_7/sub/x:output:0"gru_7/while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_7/while/gru_cell_7/mul_4Mulgru_7/while/gru_cell_7/sub:z:0)gru_7/while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/while/gru_cell_7/add_3AddV2 gru_7/while/gru_cell_7/mul_3:z:0 gru_7/while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_7_while_placeholder_1gru_7_while_placeholder gru_7/while/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_7/while/addAddV2gru_7_while_placeholdergru_7/while/add/y:output:0*
T0*
_output_shapes
: U
gru_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_7/while/add_1AddV2$gru_7_while_gru_7_while_loop_countergru_7/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_7/while/IdentityIdentitygru_7/while/add_1:z:0^gru_7/while/NoOp*
T0*
_output_shapes
: 
gru_7/while/Identity_1Identity*gru_7_while_gru_7_while_maximum_iterations^gru_7/while/NoOp*
T0*
_output_shapes
: k
gru_7/while/Identity_2Identitygru_7/while/add:z:0^gru_7/while/NoOp*
T0*
_output_shapes
: «
gru_7/while/Identity_3Identity@gru_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_7/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_7/while/Identity_4Identity gru_7/while/gru_cell_7/add_3:z:0^gru_7/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_7/while/NoOpNoOp-^gru_7/while/gru_cell_7/MatMul/ReadVariableOp/^gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp&^gru_7/while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_7_while_gru_7_strided_slice_1#gru_7_while_gru_7_strided_slice_1_0"t
7gru_7_while_gru_cell_7_matmul_1_readvariableop_resource9gru_7_while_gru_cell_7_matmul_1_readvariableop_resource_0"p
5gru_7_while_gru_cell_7_matmul_readvariableop_resource7gru_7_while_gru_cell_7_matmul_readvariableop_resource_0"b
.gru_7_while_gru_cell_7_readvariableop_resource0gru_7_while_gru_cell_7_readvariableop_resource_0"5
gru_7_while_identitygru_7/while/Identity:output:0"9
gru_7_while_identity_1gru_7/while/Identity_1:output:0"9
gru_7_while_identity_2gru_7/while/Identity_2:output:0"9
gru_7_while_identity_3gru_7/while/Identity_3:output:0"9
gru_7_while_identity_4gru_7/while/Identity_4:output:0"À
]gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_gru_7_while_tensorarrayv2read_tensorlistgetitem_gru_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_7/while/gru_cell_7/MatMul/ReadVariableOp,gru_7/while/gru_cell_7/MatMul/ReadVariableOp2`
.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp.gru_7/while/gru_cell_7/MatMul_1/ReadVariableOp2N
%gru_7/while/gru_cell_7/ReadVariableOp%gru_7/while/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_50711
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_50711___redundant_placeholder03
/while_while_cond_50711___redundant_placeholder13
/while_while_cond_50711___redundant_placeholder23
/while_while_cond_50711___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
!
Û
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51568

inputs
states_0*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51554*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdU
mul_3MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
ô3
û
@__inference_gru_7_layer_call_and_return_conditional_losses_46406

inputs#
gru_cell_7_46330:	¬#
gru_cell_7_46332:	d¬#
gru_cell_7_46334:	d¬
identity¢"gru_cell_7/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÀ
"gru_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_7_46330gru_cell_7_46332gru_cell_7_46334*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46290n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_7_46330gru_cell_7_46332gru_cell_7_46334*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46342*
condR
while_cond_46341*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿds
NoOpNoOp#^gru_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2H
"gru_cell_7/StatefulPartitionedCall"gru_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ÆQ

@__inference_gru_6_layer_call_and_return_conditional_losses_46940

inputs5
"gru_cell_6_readvariableop_resource:	¬<
)gru_cell_6_matmul_readvariableop_resource:	¬>
+gru_cell_6_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_6/MatMul/ReadVariableOp¢"gru_cell_6/MatMul_1/ReadVariableOp¢gru_cell_6/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask}
gru_cell_6/ReadVariableOpReadVariableOp"gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_6/unstackUnpack!gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_6/MatMul/ReadVariableOpReadVariableOp)gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_6/MatMulMatMulstrided_slice_2:output:0(gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAddBiasAddgru_cell_6/MatMul:product:0gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_6/splitSplit#gru_cell_6/split/split_dim:output:0gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_6/MatMul_1MatMulzeros:output:0*gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_6/BiasAdd_1BiasAddgru_cell_6/MatMul_1:product:0gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_6/split_1SplitVgru_cell_6/BiasAdd_1:output:0gru_cell_6/Const:output:0%gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_6/addAddV2gru_cell_6/split:output:0gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_6/SigmoidSigmoidgru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_6/add_1AddV2gru_cell_6/split:output:1gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_1Sigmoidgru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_6/mulMulgru_cell_6/Sigmoid_1:y:0gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_6/add_2AddV2gru_cell_6/split:output:2gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_6/mul_1Mulgru_cell_6/beta:output:0gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/Sigmoid_2Sigmoidgru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_6/mul_2Mulgru_cell_6/add_2:z:0gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_6/IdentityIdentitygru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_6/IdentityN	IdentityNgru_cell_6/mul_2:z:0gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-46828*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_6/mul_3Mulgru_cell_6/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_6/subSubgru_cell_6/sub/x:output:0gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_6/mul_4Mulgru_cell_6/sub:z:0gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_6/add_3AddV2gru_cell_6/mul_3:z:0gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_6_readvariableop_resource)gru_cell_6_matmul_readvariableop_resource+gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46844*
condR
while_cond_46843*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_6/MatMul/ReadVariableOp#^gru_cell_6/MatMul_1/ReadVariableOp^gru_cell_6/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_6/MatMul/ReadVariableOp gru_cell_6/MatMul/ReadVariableOp2H
"gru_cell_6/MatMul_1/ReadVariableOp"gru_cell_6/MatMul_1/ReadVariableOp26
gru_cell_6/ReadVariableOpgru_cell_6/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ñÅ

 __inference__wrapped_model_45711
gru_6_inputH
5sequential_2_gru_6_gru_cell_6_readvariableop_resource:	¬O
<sequential_2_gru_6_gru_cell_6_matmul_readvariableop_resource:	¬Q
>sequential_2_gru_6_gru_cell_6_matmul_1_readvariableop_resource:	d¬H
5sequential_2_gru_7_gru_cell_7_readvariableop_resource:	¬O
<sequential_2_gru_7_gru_cell_7_matmul_readvariableop_resource:	d¬Q
>sequential_2_gru_7_gru_cell_7_matmul_1_readvariableop_resource:	d¬H
5sequential_2_gru_8_gru_cell_8_readvariableop_resource:	¬O
<sequential_2_gru_8_gru_cell_8_matmul_readvariableop_resource:	d¬Q
>sequential_2_gru_8_gru_cell_8_matmul_1_readvariableop_resource:	d¬E
3sequential_2_dense_2_matmul_readvariableop_resource:dB
4sequential_2_dense_2_biasadd_readvariableop_resource:
identity¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢3sequential_2/gru_6/gru_cell_6/MatMul/ReadVariableOp¢5sequential_2/gru_6/gru_cell_6/MatMul_1/ReadVariableOp¢,sequential_2/gru_6/gru_cell_6/ReadVariableOp¢sequential_2/gru_6/while¢3sequential_2/gru_7/gru_cell_7/MatMul/ReadVariableOp¢5sequential_2/gru_7/gru_cell_7/MatMul_1/ReadVariableOp¢,sequential_2/gru_7/gru_cell_7/ReadVariableOp¢sequential_2/gru_7/while¢3sequential_2/gru_8/gru_cell_8/MatMul/ReadVariableOp¢5sequential_2/gru_8/gru_cell_8/MatMul_1/ReadVariableOp¢,sequential_2/gru_8/gru_cell_8/ReadVariableOp¢sequential_2/gru_8/whileS
sequential_2/gru_6/ShapeShapegru_6_input*
T0*
_output_shapes
:p
&sequential_2/gru_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_2/gru_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_2/gru_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_2/gru_6/strided_sliceStridedSlice!sequential_2/gru_6/Shape:output:0/sequential_2/gru_6/strided_slice/stack:output:01sequential_2/gru_6/strided_slice/stack_1:output:01sequential_2/gru_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_2/gru_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¬
sequential_2/gru_6/zeros/packedPack)sequential_2/gru_6/strided_slice:output:0*sequential_2/gru_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_2/gru_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_2/gru_6/zerosFill(sequential_2/gru_6/zeros/packed:output:0'sequential_2/gru_6/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
!sequential_2/gru_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_2/gru_6/transpose	Transposegru_6_input*sequential_2/gru_6/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿj
sequential_2/gru_6/Shape_1Shape sequential_2/gru_6/transpose:y:0*
T0*
_output_shapes
:r
(sequential_2/gru_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_2/gru_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_2/gru_6/strided_slice_1StridedSlice#sequential_2/gru_6/Shape_1:output:01sequential_2/gru_6/strided_slice_1/stack:output:03sequential_2/gru_6/strided_slice_1/stack_1:output:03sequential_2/gru_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_2/gru_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_2/gru_6/TensorArrayV2TensorListReserve7sequential_2/gru_6/TensorArrayV2/element_shape:output:0+sequential_2/gru_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_2/gru_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
:sequential_2/gru_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_2/gru_6/transpose:y:0Qsequential_2/gru_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_2/gru_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_2/gru_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_2/gru_6/strided_slice_2StridedSlice sequential_2/gru_6/transpose:y:01sequential_2/gru_6/strided_slice_2/stack:output:03sequential_2/gru_6/strided_slice_2/stack_1:output:03sequential_2/gru_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask£
,sequential_2/gru_6/gru_cell_6/ReadVariableOpReadVariableOp5sequential_2_gru_6_gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0
%sequential_2/gru_6/gru_cell_6/unstackUnpack4sequential_2/gru_6/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num±
3sequential_2/gru_6/gru_cell_6/MatMul/ReadVariableOpReadVariableOp<sequential_2_gru_6_gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0Ë
$sequential_2/gru_6/gru_cell_6/MatMulMatMul+sequential_2/gru_6/strided_slice_2:output:0;sequential_2/gru_6/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
%sequential_2/gru_6/gru_cell_6/BiasAddBiasAdd.sequential_2/gru_6/gru_cell_6/MatMul:product:0.sequential_2/gru_6/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
-sequential_2/gru_6/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#sequential_2/gru_6/gru_cell_6/splitSplit6sequential_2/gru_6/gru_cell_6/split/split_dim:output:0.sequential_2/gru_6/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitµ
5sequential_2/gru_6/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp>sequential_2_gru_6_gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Å
&sequential_2/gru_6/gru_cell_6/MatMul_1MatMul!sequential_2/gru_6/zeros:output:0=sequential_2/gru_6/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ç
'sequential_2/gru_6/gru_cell_6/BiasAdd_1BiasAdd0sequential_2/gru_6/gru_cell_6/MatMul_1:product:0.sequential_2/gru_6/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
#sequential_2/gru_6/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿz
/sequential_2/gru_6/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%sequential_2/gru_6/gru_cell_6/split_1SplitV0sequential_2/gru_6/gru_cell_6/BiasAdd_1:output:0,sequential_2/gru_6/gru_cell_6/Const:output:08sequential_2/gru_6/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitº
!sequential_2/gru_6/gru_cell_6/addAddV2,sequential_2/gru_6/gru_cell_6/split:output:0.sequential_2/gru_6/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%sequential_2/gru_6/gru_cell_6/SigmoidSigmoid%sequential_2/gru_6/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¼
#sequential_2/gru_6/gru_cell_6/add_1AddV2,sequential_2/gru_6/gru_cell_6/split:output:1.sequential_2/gru_6/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_2/gru_6/gru_cell_6/Sigmoid_1Sigmoid'sequential_2/gru_6/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd·
!sequential_2/gru_6/gru_cell_6/mulMul+sequential_2/gru_6/gru_cell_6/Sigmoid_1:y:0.sequential_2/gru_6/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
#sequential_2/gru_6/gru_cell_6/add_2AddV2,sequential_2/gru_6/gru_cell_6/split:output:2%sequential_2/gru_6/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"sequential_2/gru_6/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
#sequential_2/gru_6/gru_cell_6/mul_1Mul+sequential_2/gru_6/gru_cell_6/beta:output:0'sequential_2/gru_6/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_2/gru_6/gru_cell_6/Sigmoid_2Sigmoid'sequential_2/gru_6/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
#sequential_2/gru_6/gru_cell_6/mul_2Mul'sequential_2/gru_6/gru_cell_6/add_2:z:0+sequential_2/gru_6/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_2/gru_6/gru_cell_6/IdentityIdentity'sequential_2/gru_6/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdü
'sequential_2/gru_6/gru_cell_6/IdentityN	IdentityN'sequential_2/gru_6/gru_cell_6/mul_2:z:0'sequential_2/gru_6/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45267*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdª
#sequential_2/gru_6/gru_cell_6/mul_3Mul)sequential_2/gru_6/gru_cell_6/Sigmoid:y:0!sequential_2/gru_6/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential_2/gru_6/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential_2/gru_6/gru_cell_6/subSub,sequential_2/gru_6/gru_cell_6/sub/x:output:0)sequential_2/gru_6/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
#sequential_2/gru_6/gru_cell_6/mul_4Mul%sequential_2/gru_6/gru_cell_6/sub:z:00sequential_2/gru_6/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
#sequential_2/gru_6/gru_cell_6/add_3AddV2'sequential_2/gru_6/gru_cell_6/mul_3:z:0'sequential_2/gru_6/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential_2/gru_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ñ
"sequential_2/gru_6/TensorArrayV2_1TensorListReserve9sequential_2/gru_6/TensorArrayV2_1/element_shape:output:0+sequential_2/gru_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_2/gru_6/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_2/gru_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_2/gru_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
sequential_2/gru_6/whileWhile.sequential_2/gru_6/while/loop_counter:output:04sequential_2/gru_6/while/maximum_iterations:output:0 sequential_2/gru_6/time:output:0+sequential_2/gru_6/TensorArrayV2_1:handle:0!sequential_2/gru_6/zeros:output:0+sequential_2/gru_6/strided_slice_1:output:0Jsequential_2/gru_6/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_2_gru_6_gru_cell_6_readvariableop_resource<sequential_2_gru_6_gru_cell_6_matmul_readvariableop_resource>sequential_2_gru_6_gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( */
body'R%
#sequential_2_gru_6_while_body_45283*/
cond'R%
#sequential_2_gru_6_while_cond_45282*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Csequential_2/gru_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   û
5sequential_2/gru_6/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_2/gru_6/while:output:3Lsequential_2/gru_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0{
(sequential_2/gru_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_2/gru_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_2/gru_6/strided_slice_3StridedSlice>sequential_2/gru_6/TensorArrayV2Stack/TensorListStack:tensor:01sequential_2/gru_6/strided_slice_3/stack:output:03sequential_2/gru_6/strided_slice_3/stack_1:output:03sequential_2/gru_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskx
#sequential_2/gru_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_2/gru_6/transpose_1	Transpose>sequential_2/gru_6/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_2/gru_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿddn
sequential_2/gru_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
sequential_2/gru_7/ShapeShape"sequential_2/gru_6/transpose_1:y:0*
T0*
_output_shapes
:p
&sequential_2/gru_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_2/gru_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_2/gru_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_2/gru_7/strided_sliceStridedSlice!sequential_2/gru_7/Shape:output:0/sequential_2/gru_7/strided_slice/stack:output:01sequential_2/gru_7/strided_slice/stack_1:output:01sequential_2/gru_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_2/gru_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¬
sequential_2/gru_7/zeros/packedPack)sequential_2/gru_7/strided_slice:output:0*sequential_2/gru_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_2/gru_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_2/gru_7/zerosFill(sequential_2/gru_7/zeros/packed:output:0'sequential_2/gru_7/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
!sequential_2/gru_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
sequential_2/gru_7/transpose	Transpose"sequential_2/gru_6/transpose_1:y:0*sequential_2/gru_7/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdj
sequential_2/gru_7/Shape_1Shape sequential_2/gru_7/transpose:y:0*
T0*
_output_shapes
:r
(sequential_2/gru_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_2/gru_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_2/gru_7/strided_slice_1StridedSlice#sequential_2/gru_7/Shape_1:output:01sequential_2/gru_7/strided_slice_1/stack:output:03sequential_2/gru_7/strided_slice_1/stack_1:output:03sequential_2/gru_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_2/gru_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_2/gru_7/TensorArrayV2TensorListReserve7sequential_2/gru_7/TensorArrayV2/element_shape:output:0+sequential_2/gru_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_2/gru_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
:sequential_2/gru_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_2/gru_7/transpose:y:0Qsequential_2/gru_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_2/gru_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_2/gru_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_2/gru_7/strided_slice_2StridedSlice sequential_2/gru_7/transpose:y:01sequential_2/gru_7/strided_slice_2/stack:output:03sequential_2/gru_7/strided_slice_2/stack_1:output:03sequential_2/gru_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask£
,sequential_2/gru_7/gru_cell_7/ReadVariableOpReadVariableOp5sequential_2_gru_7_gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0
%sequential_2/gru_7/gru_cell_7/unstackUnpack4sequential_2/gru_7/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num±
3sequential_2/gru_7/gru_cell_7/MatMul/ReadVariableOpReadVariableOp<sequential_2_gru_7_gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Ë
$sequential_2/gru_7/gru_cell_7/MatMulMatMul+sequential_2/gru_7/strided_slice_2:output:0;sequential_2/gru_7/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
%sequential_2/gru_7/gru_cell_7/BiasAddBiasAdd.sequential_2/gru_7/gru_cell_7/MatMul:product:0.sequential_2/gru_7/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
-sequential_2/gru_7/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#sequential_2/gru_7/gru_cell_7/splitSplit6sequential_2/gru_7/gru_cell_7/split/split_dim:output:0.sequential_2/gru_7/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitµ
5sequential_2/gru_7/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp>sequential_2_gru_7_gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Å
&sequential_2/gru_7/gru_cell_7/MatMul_1MatMul!sequential_2/gru_7/zeros:output:0=sequential_2/gru_7/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ç
'sequential_2/gru_7/gru_cell_7/BiasAdd_1BiasAdd0sequential_2/gru_7/gru_cell_7/MatMul_1:product:0.sequential_2/gru_7/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
#sequential_2/gru_7/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿz
/sequential_2/gru_7/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%sequential_2/gru_7/gru_cell_7/split_1SplitV0sequential_2/gru_7/gru_cell_7/BiasAdd_1:output:0,sequential_2/gru_7/gru_cell_7/Const:output:08sequential_2/gru_7/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitº
!sequential_2/gru_7/gru_cell_7/addAddV2,sequential_2/gru_7/gru_cell_7/split:output:0.sequential_2/gru_7/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%sequential_2/gru_7/gru_cell_7/SigmoidSigmoid%sequential_2/gru_7/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¼
#sequential_2/gru_7/gru_cell_7/add_1AddV2,sequential_2/gru_7/gru_cell_7/split:output:1.sequential_2/gru_7/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_2/gru_7/gru_cell_7/Sigmoid_1Sigmoid'sequential_2/gru_7/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd·
!sequential_2/gru_7/gru_cell_7/mulMul+sequential_2/gru_7/gru_cell_7/Sigmoid_1:y:0.sequential_2/gru_7/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
#sequential_2/gru_7/gru_cell_7/add_2AddV2,sequential_2/gru_7/gru_cell_7/split:output:2%sequential_2/gru_7/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"sequential_2/gru_7/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
#sequential_2/gru_7/gru_cell_7/mul_1Mul+sequential_2/gru_7/gru_cell_7/beta:output:0'sequential_2/gru_7/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_2/gru_7/gru_cell_7/Sigmoid_2Sigmoid'sequential_2/gru_7/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
#sequential_2/gru_7/gru_cell_7/mul_2Mul'sequential_2/gru_7/gru_cell_7/add_2:z:0+sequential_2/gru_7/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_2/gru_7/gru_cell_7/IdentityIdentity'sequential_2/gru_7/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdü
'sequential_2/gru_7/gru_cell_7/IdentityN	IdentityN'sequential_2/gru_7/gru_cell_7/mul_2:z:0'sequential_2/gru_7/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45430*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdª
#sequential_2/gru_7/gru_cell_7/mul_3Mul)sequential_2/gru_7/gru_cell_7/Sigmoid:y:0!sequential_2/gru_7/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential_2/gru_7/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential_2/gru_7/gru_cell_7/subSub,sequential_2/gru_7/gru_cell_7/sub/x:output:0)sequential_2/gru_7/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
#sequential_2/gru_7/gru_cell_7/mul_4Mul%sequential_2/gru_7/gru_cell_7/sub:z:00sequential_2/gru_7/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
#sequential_2/gru_7/gru_cell_7/add_3AddV2'sequential_2/gru_7/gru_cell_7/mul_3:z:0'sequential_2/gru_7/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential_2/gru_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ñ
"sequential_2/gru_7/TensorArrayV2_1TensorListReserve9sequential_2/gru_7/TensorArrayV2_1/element_shape:output:0+sequential_2/gru_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_2/gru_7/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_2/gru_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_2/gru_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
sequential_2/gru_7/whileWhile.sequential_2/gru_7/while/loop_counter:output:04sequential_2/gru_7/while/maximum_iterations:output:0 sequential_2/gru_7/time:output:0+sequential_2/gru_7/TensorArrayV2_1:handle:0!sequential_2/gru_7/zeros:output:0+sequential_2/gru_7/strided_slice_1:output:0Jsequential_2/gru_7/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_2_gru_7_gru_cell_7_readvariableop_resource<sequential_2_gru_7_gru_cell_7_matmul_readvariableop_resource>sequential_2_gru_7_gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( */
body'R%
#sequential_2_gru_7_while_body_45446*/
cond'R%
#sequential_2_gru_7_while_cond_45445*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Csequential_2/gru_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   û
5sequential_2/gru_7/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_2/gru_7/while:output:3Lsequential_2/gru_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0{
(sequential_2/gru_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_2/gru_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_2/gru_7/strided_slice_3StridedSlice>sequential_2/gru_7/TensorArrayV2Stack/TensorListStack:tensor:01sequential_2/gru_7/strided_slice_3/stack:output:03sequential_2/gru_7/strided_slice_3/stack_1:output:03sequential_2/gru_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskx
#sequential_2/gru_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_2/gru_7/transpose_1	Transpose>sequential_2/gru_7/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_2/gru_7/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿddn
sequential_2/gru_7/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
sequential_2/gru_8/ShapeShape"sequential_2/gru_7/transpose_1:y:0*
T0*
_output_shapes
:p
&sequential_2/gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_2/gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_2/gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_2/gru_8/strided_sliceStridedSlice!sequential_2/gru_8/Shape:output:0/sequential_2/gru_8/strided_slice/stack:output:01sequential_2/gru_8/strided_slice/stack_1:output:01sequential_2/gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_2/gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¬
sequential_2/gru_8/zeros/packedPack)sequential_2/gru_8/strided_slice:output:0*sequential_2/gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_2/gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_2/gru_8/zerosFill(sequential_2/gru_8/zeros/packed:output:0'sequential_2/gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
!sequential_2/gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
sequential_2/gru_8/transpose	Transpose"sequential_2/gru_7/transpose_1:y:0*sequential_2/gru_8/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdj
sequential_2/gru_8/Shape_1Shape sequential_2/gru_8/transpose:y:0*
T0*
_output_shapes
:r
(sequential_2/gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_2/gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_2/gru_8/strided_slice_1StridedSlice#sequential_2/gru_8/Shape_1:output:01sequential_2/gru_8/strided_slice_1/stack:output:03sequential_2/gru_8/strided_slice_1/stack_1:output:03sequential_2/gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_2/gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_2/gru_8/TensorArrayV2TensorListReserve7sequential_2/gru_8/TensorArrayV2/element_shape:output:0+sequential_2/gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_2/gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
:sequential_2/gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_2/gru_8/transpose:y:0Qsequential_2/gru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_2/gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_2/gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_2/gru_8/strided_slice_2StridedSlice sequential_2/gru_8/transpose:y:01sequential_2/gru_8/strided_slice_2/stack:output:03sequential_2/gru_8/strided_slice_2/stack_1:output:03sequential_2/gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask£
,sequential_2/gru_8/gru_cell_8/ReadVariableOpReadVariableOp5sequential_2_gru_8_gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0
%sequential_2/gru_8/gru_cell_8/unstackUnpack4sequential_2/gru_8/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num±
3sequential_2/gru_8/gru_cell_8/MatMul/ReadVariableOpReadVariableOp<sequential_2_gru_8_gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Ë
$sequential_2/gru_8/gru_cell_8/MatMulMatMul+sequential_2/gru_8/strided_slice_2:output:0;sequential_2/gru_8/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
%sequential_2/gru_8/gru_cell_8/BiasAddBiasAdd.sequential_2/gru_8/gru_cell_8/MatMul:product:0.sequential_2/gru_8/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
-sequential_2/gru_8/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#sequential_2/gru_8/gru_cell_8/splitSplit6sequential_2/gru_8/gru_cell_8/split/split_dim:output:0.sequential_2/gru_8/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitµ
5sequential_2/gru_8/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp>sequential_2_gru_8_gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Å
&sequential_2/gru_8/gru_cell_8/MatMul_1MatMul!sequential_2/gru_8/zeros:output:0=sequential_2/gru_8/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ç
'sequential_2/gru_8/gru_cell_8/BiasAdd_1BiasAdd0sequential_2/gru_8/gru_cell_8/MatMul_1:product:0.sequential_2/gru_8/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
#sequential_2/gru_8/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿz
/sequential_2/gru_8/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%sequential_2/gru_8/gru_cell_8/split_1SplitV0sequential_2/gru_8/gru_cell_8/BiasAdd_1:output:0,sequential_2/gru_8/gru_cell_8/Const:output:08sequential_2/gru_8/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitº
!sequential_2/gru_8/gru_cell_8/addAddV2,sequential_2/gru_8/gru_cell_8/split:output:0.sequential_2/gru_8/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%sequential_2/gru_8/gru_cell_8/SigmoidSigmoid%sequential_2/gru_8/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¼
#sequential_2/gru_8/gru_cell_8/add_1AddV2,sequential_2/gru_8/gru_cell_8/split:output:1.sequential_2/gru_8/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_2/gru_8/gru_cell_8/Sigmoid_1Sigmoid'sequential_2/gru_8/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd·
!sequential_2/gru_8/gru_cell_8/mulMul+sequential_2/gru_8/gru_cell_8/Sigmoid_1:y:0.sequential_2/gru_8/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
#sequential_2/gru_8/gru_cell_8/add_2AddV2,sequential_2/gru_8/gru_cell_8/split:output:2%sequential_2/gru_8/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"sequential_2/gru_8/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
#sequential_2/gru_8/gru_cell_8/mul_1Mul+sequential_2/gru_8/gru_cell_8/beta:output:0'sequential_2/gru_8/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_2/gru_8/gru_cell_8/Sigmoid_2Sigmoid'sequential_2/gru_8/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
#sequential_2/gru_8/gru_cell_8/mul_2Mul'sequential_2/gru_8/gru_cell_8/add_2:z:0+sequential_2/gru_8/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_2/gru_8/gru_cell_8/IdentityIdentity'sequential_2/gru_8/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdü
'sequential_2/gru_8/gru_cell_8/IdentityN	IdentityN'sequential_2/gru_8/gru_cell_8/mul_2:z:0'sequential_2/gru_8/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45593*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdª
#sequential_2/gru_8/gru_cell_8/mul_3Mul)sequential_2/gru_8/gru_cell_8/Sigmoid:y:0!sequential_2/gru_8/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential_2/gru_8/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential_2/gru_8/gru_cell_8/subSub,sequential_2/gru_8/gru_cell_8/sub/x:output:0)sequential_2/gru_8/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
#sequential_2/gru_8/gru_cell_8/mul_4Mul%sequential_2/gru_8/gru_cell_8/sub:z:00sequential_2/gru_8/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
#sequential_2/gru_8/gru_cell_8/add_3AddV2'sequential_2/gru_8/gru_cell_8/mul_3:z:0'sequential_2/gru_8/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential_2/gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ñ
"sequential_2/gru_8/TensorArrayV2_1TensorListReserve9sequential_2/gru_8/TensorArrayV2_1/element_shape:output:0+sequential_2/gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_2/gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_2/gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_2/gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
sequential_2/gru_8/whileWhile.sequential_2/gru_8/while/loop_counter:output:04sequential_2/gru_8/while/maximum_iterations:output:0 sequential_2/gru_8/time:output:0+sequential_2/gru_8/TensorArrayV2_1:handle:0!sequential_2/gru_8/zeros:output:0+sequential_2/gru_8/strided_slice_1:output:0Jsequential_2/gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_2_gru_8_gru_cell_8_readvariableop_resource<sequential_2_gru_8_gru_cell_8_matmul_readvariableop_resource>sequential_2_gru_8_gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( */
body'R%
#sequential_2_gru_8_while_body_45609*/
cond'R%
#sequential_2_gru_8_while_cond_45608*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Csequential_2/gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   û
5sequential_2/gru_8/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_2/gru_8/while:output:3Lsequential_2/gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0{
(sequential_2/gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_2/gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_2/gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_2/gru_8/strided_slice_3StridedSlice>sequential_2/gru_8/TensorArrayV2Stack/TensorListStack:tensor:01sequential_2/gru_8/strided_slice_3/stack:output:03sequential_2/gru_8/strided_slice_3/stack_1:output:03sequential_2/gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskx
#sequential_2/gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_2/gru_8/transpose_1	Transpose>sequential_2/gru_8/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_2/gru_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿddn
sequential_2/gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0¸
sequential_2/dense_2/MatMulMatMul+sequential_2/gru_8/strided_slice_3:output:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%sequential_2/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
NoOpNoOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp4^sequential_2/gru_6/gru_cell_6/MatMul/ReadVariableOp6^sequential_2/gru_6/gru_cell_6/MatMul_1/ReadVariableOp-^sequential_2/gru_6/gru_cell_6/ReadVariableOp^sequential_2/gru_6/while4^sequential_2/gru_7/gru_cell_7/MatMul/ReadVariableOp6^sequential_2/gru_7/gru_cell_7/MatMul_1/ReadVariableOp-^sequential_2/gru_7/gru_cell_7/ReadVariableOp^sequential_2/gru_7/while4^sequential_2/gru_8/gru_cell_8/MatMul/ReadVariableOp6^sequential_2/gru_8/gru_cell_8/MatMul_1/ReadVariableOp-^sequential_2/gru_8/gru_cell_8/ReadVariableOp^sequential_2/gru_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2j
3sequential_2/gru_6/gru_cell_6/MatMul/ReadVariableOp3sequential_2/gru_6/gru_cell_6/MatMul/ReadVariableOp2n
5sequential_2/gru_6/gru_cell_6/MatMul_1/ReadVariableOp5sequential_2/gru_6/gru_cell_6/MatMul_1/ReadVariableOp2\
,sequential_2/gru_6/gru_cell_6/ReadVariableOp,sequential_2/gru_6/gru_cell_6/ReadVariableOp24
sequential_2/gru_6/whilesequential_2/gru_6/while2j
3sequential_2/gru_7/gru_cell_7/MatMul/ReadVariableOp3sequential_2/gru_7/gru_cell_7/MatMul/ReadVariableOp2n
5sequential_2/gru_7/gru_cell_7/MatMul_1/ReadVariableOp5sequential_2/gru_7/gru_cell_7/MatMul_1/ReadVariableOp2\
,sequential_2/gru_7/gru_cell_7/ReadVariableOp,sequential_2/gru_7/gru_cell_7/ReadVariableOp24
sequential_2/gru_7/whilesequential_2/gru_7/while2j
3sequential_2/gru_8/gru_cell_8/MatMul/ReadVariableOp3sequential_2/gru_8/gru_cell_8/MatMul/ReadVariableOp2n
5sequential_2/gru_8/gru_cell_8/MatMul_1/ReadVariableOp5sequential_2/gru_8/gru_cell_8/MatMul_1/ReadVariableOp2\
,sequential_2/gru_8/gru_cell_8/ReadVariableOp,sequential_2/gru_8/gru_cell_8/ReadVariableOp24
sequential_2/gru_8/whilesequential_2/gru_8/while:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_6_input
!
Û
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51448

inputs
states_0*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51434*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdU
mul_3MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
!
Ù
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45938

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45924*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdS
mul_3MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
ý

"__inference_internal_grad_fn_52240
result_grads_0
result_grads_1
mul_gru_7_gru_cell_7_beta
mul_gru_7_gru_cell_7_add_2
identity
mulMulmul_gru_7_gru_cell_7_betamul_gru_7_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_7_gru_cell_7_betamul_gru_7_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ô

¬
,__inference_sequential_2_layer_call_fn_47338
gru_6_input
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
	unknown_2:	¬
	unknown_3:	d¬
	unknown_4:	d¬
	unknown_5:	¬
	unknown_6:	d¬
	unknown_7:	d¬
	unknown_8:d
	unknown_9:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallgru_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_47313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_6_input
Õ
¥
while_cond_46693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46693___redundant_placeholder03
/while_while_cond_46693___redundant_placeholder13
/while_while_cond_46693___redundant_placeholder23
/while_while_cond_46693___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ø

"__inference_internal_grad_fn_52582
result_grads_0
result_grads_1
mul_gru_cell_7_beta
mul_gru_cell_7_add_2
identityx
mulMulmul_gru_cell_7_betamul_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_7_betamul_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý

"__inference_internal_grad_fn_52456
result_grads_0
result_grads_1
mul_while_gru_cell_6_beta
mul_while_gru_cell_6_add_2
identity
mulMulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
¥
while_cond_46152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46152___redundant_placeholder03
/while_while_cond_46152___redundant_placeholder13
/while_while_cond_46152___redundant_placeholder23
/while_while_cond_46152___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ô3
û
@__inference_gru_7_layer_call_and_return_conditional_losses_46217

inputs#
gru_cell_7_46141:	¬#
gru_cell_7_46143:	d¬#
gru_cell_7_46145:	d¬
identity¢"gru_cell_7/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskÀ
"gru_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_7_46141gru_cell_7_46143gru_cell_7_46145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46140n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_7_46141gru_cell_7_46143gru_cell_7_46145*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46153*
condR
while_cond_46152*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿds
NoOpNoOp#^gru_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2H
"gru_cell_7/StatefulPartitionedCall"gru_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Õ
¥
while_cond_46843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46843___redundant_placeholder03
/while_while_cond_46843___redundant_placeholder13
/while_while_cond_46843___redundant_placeholder23
/while_while_cond_46843___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
B
þ
while_body_51046
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_8_readvariableop_resource_0:	¬D
1while_gru_cell_8_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_8_readvariableop_resource:	¬B
/while_gru_cell_8_matmul_readvariableop_resource:	d¬D
1while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_8/MatMul/ReadVariableOp¢(while/gru_cell_8/MatMul_1/ReadVariableOp¢while/gru_cell_8/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_8/ReadVariableOpReadVariableOp*while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_8/unstackUnpack'while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_8/BiasAddBiasAdd!while/gru_cell_8/MatMul:product:0!while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_8/splitSplit)while/gru_cell_8/split/split_dim:output:0!while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_8/MatMul_1MatMulwhile_placeholder_20while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_8/BiasAdd_1BiasAdd#while/gru_cell_8/MatMul_1:product:0!while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_8/split_1SplitV#while/gru_cell_8/BiasAdd_1:output:0while/gru_cell_8/Const:output:0+while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_8/addAddV2while/gru_cell_8/split:output:0!while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_8/SigmoidSigmoidwhile/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_1AddV2while/gru_cell_8/split:output:1!while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_1Sigmoidwhile/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mulMulwhile/gru_cell_8/Sigmoid_1:y:0!while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_2AddV2while/gru_cell_8/split:output:2while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/mul_1Mulwhile/gru_cell_8/beta:output:0while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_2Sigmoidwhile/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_2Mulwhile/gru_cell_8/add_2:z:0while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/IdentityIdentitywhile/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_8/IdentityN	IdentityNwhile/gru_cell_8/mul_2:z:0while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51096*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_3Mulwhile/gru_cell_8/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/subSubwhile/gru_cell_8/sub/x:output:0while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_4Mulwhile/gru_cell_8/sub:z:0#while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_3AddV2while/gru_cell_8/mul_3:z:0while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_8/MatMul/ReadVariableOp)^while/gru_cell_8/MatMul_1/ReadVariableOp ^while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_8_matmul_1_readvariableop_resource3while_gru_cell_8_matmul_1_readvariableop_resource_0"d
/while_gru_cell_8_matmul_readvariableop_resource1while_gru_cell_8_matmul_readvariableop_resource_0"V
(while_gru_cell_8_readvariableop_resource*while_gru_cell_8_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_8/MatMul/ReadVariableOp&while/gru_cell_8/MatMul/ReadVariableOp2T
(while/gru_cell_8/MatMul_1/ReadVariableOp(while/gru_cell_8/MatMul_1/ReadVariableOp2B
while/gru_cell_8/ReadVariableOpwhile/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
R

@__inference_gru_8_layer_call_and_return_conditional_losses_50975
inputs_05
"gru_cell_8_readvariableop_resource:	¬<
)gru_cell_8_matmul_readvariableop_resource:	d¬>
+gru_cell_8_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_8/MatMul/ReadVariableOp¢"gru_cell_8/MatMul_1/ReadVariableOp¢gru_cell_8/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_8/ReadVariableOpReadVariableOp"gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_8/unstackUnpack!gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_8/MatMul/ReadVariableOpReadVariableOp)gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMulMatMulstrided_slice_2:output:0(gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAddBiasAddgru_cell_8/MatMul:product:0gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_8/splitSplit#gru_cell_8/split/split_dim:output:0gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMul_1MatMulzeros:output:0*gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAdd_1BiasAddgru_cell_8/MatMul_1:product:0gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_8/split_1SplitVgru_cell_8/BiasAdd_1:output:0gru_cell_8/Const:output:0%gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_8/addAddV2gru_cell_8/split:output:0gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_8/SigmoidSigmoidgru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_8/add_1AddV2gru_cell_8/split:output:1gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_1Sigmoidgru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_8/mulMulgru_cell_8/Sigmoid_1:y:0gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_8/add_2AddV2gru_cell_8/split:output:2gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_8/mul_1Mulgru_cell_8/beta:output:0gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_2Sigmoidgru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_8/mul_2Mulgru_cell_8/add_2:z:0gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/IdentityIdentitygru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_8/IdentityN	IdentityNgru_cell_8/mul_2:z:0gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50863*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_8/mul_3Mulgru_cell_8/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_8/subSubgru_cell_8/sub/x:output:0gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_8/mul_4Mulgru_cell_8/sub:z:0gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_8/add_3AddV2gru_cell_8/mul_3:z:0gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_8_readvariableop_resource)gru_cell_8_matmul_readvariableop_resource+gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_50879*
condR
while_cond_50878*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_8/MatMul/ReadVariableOp#^gru_cell_8/MatMul_1/ReadVariableOp^gru_cell_8/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_8/MatMul/ReadVariableOp gru_cell_8/MatMul/ReadVariableOp2H
"gru_cell_8/MatMul_1/ReadVariableOp"gru_cell_8/MatMul_1/ReadVariableOp26
gru_cell_8/ReadVariableOpgru_cell_8/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
ý

"__inference_internal_grad_fn_52672
result_grads_0
result_grads_1
mul_while_gru_cell_7_beta
mul_while_gru_cell_7_add_2
identity
mulMulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
å
§
while_body_45801
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_6_45823_0:	¬+
while_gru_cell_6_45825_0:	¬+
while_gru_cell_6_45827_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_6_45823:	¬)
while_gru_cell_6_45825:	¬)
while_gru_cell_6_45827:	d¬¢(while/gru_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_6_45823_0while_gru_cell_6_45825_0while_gru_cell_6_45827_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45788Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_6_45823while_gru_cell_6_45823_0"2
while_gru_cell_6_45825while_gru_cell_6_45825_0"2
while_gru_cell_6_45827while_gru_cell_6_45827_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/gru_cell_6/StatefulPartitionedCall(while/gru_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý

"__inference_internal_grad_fn_52852
result_grads_0
result_grads_1
mul_while_gru_cell_8_beta
mul_while_gru_cell_8_add_2
identity
mulMulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ø

"__inference_internal_grad_fn_52762
result_grads_0
result_grads_1
mul_gru_cell_8_beta
mul_gru_cell_8_add_2
identityx
mulMulmul_gru_cell_8_betamul_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_8_betamul_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý
¶
%__inference_gru_7_layer_call_fn_49929

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_47717s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ý

"__inference_internal_grad_fn_52780
result_grads_0
result_grads_1
mul_while_gru_cell_8_beta
mul_while_gru_cell_8_add_2
identity
mulMulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

w
"__inference_internal_grad_fn_52942
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý

"__inference_internal_grad_fn_51988
result_grads_0
result_grads_1
mul_while_gru_cell_8_beta
mul_while_gru_cell_8_add_2
identity
mulMulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_50167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_7_readvariableop_resource_0:	¬D
1while_gru_cell_7_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_7_readvariableop_resource:	¬B
/while_gru_cell_7_matmul_readvariableop_resource:	d¬D
1while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_7/MatMul/ReadVariableOp¢(while/gru_cell_7/MatMul_1/ReadVariableOp¢while/gru_cell_7/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_7/ReadVariableOpReadVariableOp*while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_7/unstackUnpack'while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_7/BiasAddBiasAdd!while/gru_cell_7/MatMul:product:0!while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_7/splitSplit)while/gru_cell_7/split/split_dim:output:0!while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_7/MatMul_1MatMulwhile_placeholder_20while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_7/BiasAdd_1BiasAdd#while/gru_cell_7/MatMul_1:product:0!while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_7/split_1SplitV#while/gru_cell_7/BiasAdd_1:output:0while/gru_cell_7/Const:output:0+while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_7/addAddV2while/gru_cell_7/split:output:0!while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_7/SigmoidSigmoidwhile/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_1AddV2while/gru_cell_7/split:output:1!while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_1Sigmoidwhile/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mulMulwhile/gru_cell_7/Sigmoid_1:y:0!while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_2AddV2while/gru_cell_7/split:output:2while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/mul_1Mulwhile/gru_cell_7/beta:output:0while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_2Sigmoidwhile/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_2Mulwhile/gru_cell_7/add_2:z:0while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/IdentityIdentitywhile/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_7/IdentityN	IdentityNwhile/gru_cell_7/mul_2:z:0while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50217*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_3Mulwhile/gru_cell_7/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/subSubwhile/gru_cell_7/sub/x:output:0while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_4Mulwhile/gru_cell_7/sub:z:0#while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_3AddV2while/gru_cell_7/mul_3:z:0while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_7/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_7/MatMul/ReadVariableOp)^while/gru_cell_7/MatMul_1/ReadVariableOp ^while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_7_matmul_1_readvariableop_resource3while_gru_cell_7_matmul_1_readvariableop_resource_0"d
/while_gru_cell_7_matmul_readvariableop_resource1while_gru_cell_7_matmul_readvariableop_resource_0"V
(while_gru_cell_7_readvariableop_resource*while_gru_cell_7_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_7/MatMul/ReadVariableOp&while/gru_cell_7/MatMul/ReadVariableOp2T
(while/gru_cell_7/MatMul_1/ReadVariableOp(while/gru_cell_7/MatMul_1/ReadVariableOp2B
while/gru_cell_7/ReadVariableOpwhile/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 

¸
%__inference_gru_8_layer_call_fn_50619
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_46758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
å
§
while_body_46153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_7_46175_0:	¬+
while_gru_cell_7_46177_0:	d¬+
while_gru_cell_7_46179_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_7_46175:	¬)
while_gru_cell_7_46177:	d¬)
while_gru_cell_7_46179:	d¬¢(while/gru_cell_7/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0û
(while/gru_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_7_46175_0while_gru_cell_7_46177_0while_gru_cell_7_46179_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46140Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_7_46175while_gru_cell_7_46175_0"2
while_gru_cell_7_46177while_gru_cell_7_46177_0"2
while_gru_cell_7_46179while_gru_cell_7_46179_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/gru_cell_7/StatefulPartitionedCall(while/gru_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_51212
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_51212___redundant_placeholder03
/while_while_cond_51212___redundant_placeholder13
/while_while_cond_51212___redundant_placeholder23
/while_while_cond_51212___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:

w
"__inference_internal_grad_fn_52870
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
J
²	
gru_6_while_body_48217(
$gru_6_while_gru_6_while_loop_counter.
*gru_6_while_gru_6_while_maximum_iterations
gru_6_while_placeholder
gru_6_while_placeholder_1
gru_6_while_placeholder_2'
#gru_6_while_gru_6_strided_slice_1_0c
_gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_0C
0gru_6_while_gru_cell_6_readvariableop_resource_0:	¬J
7gru_6_while_gru_cell_6_matmul_readvariableop_resource_0:	¬L
9gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
gru_6_while_identity
gru_6_while_identity_1
gru_6_while_identity_2
gru_6_while_identity_3
gru_6_while_identity_4%
!gru_6_while_gru_6_strided_slice_1a
]gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensorA
.gru_6_while_gru_cell_6_readvariableop_resource:	¬H
5gru_6_while_gru_cell_6_matmul_readvariableop_resource:	¬J
7gru_6_while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢,gru_6/while/gru_cell_6/MatMul/ReadVariableOp¢.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp¢%gru_6/while/gru_cell_6/ReadVariableOp
=gru_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/gru_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_0gru_6_while_placeholderFgru_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%gru_6/while/gru_cell_6/ReadVariableOpReadVariableOp0gru_6_while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_6/while/gru_cell_6/unstackUnpack-gru_6/while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_6/while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp7gru_6_while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0È
gru_6/while/gru_cell_6/MatMulMatMul6gru_6/while/TensorArrayV2Read/TensorListGetItem:item:04gru_6/while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_6/while/gru_cell_6/BiasAddBiasAdd'gru_6/while/gru_cell_6/MatMul:product:0'gru_6/while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_6/while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_6/while/gru_cell_6/splitSplit/gru_6/while/gru_cell_6/split/split_dim:output:0'gru_6/while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp9gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_6/while/gru_cell_6/MatMul_1MatMulgru_6_while_placeholder_26gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_6/while/gru_cell_6/BiasAdd_1BiasAdd)gru_6/while/gru_cell_6/MatMul_1:product:0'gru_6/while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_6/while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_6/while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_6/while/gru_cell_6/split_1SplitV)gru_6/while/gru_cell_6/BiasAdd_1:output:0%gru_6/while/gru_cell_6/Const:output:01gru_6/while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_6/while/gru_cell_6/addAddV2%gru_6/while/gru_cell_6/split:output:0'gru_6/while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_6/while/gru_cell_6/SigmoidSigmoidgru_6/while/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_6/while/gru_cell_6/add_1AddV2%gru_6/while/gru_cell_6/split:output:1'gru_6/while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_6/while/gru_cell_6/Sigmoid_1Sigmoid gru_6/while/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_6/while/gru_cell_6/mulMul$gru_6/while/gru_cell_6/Sigmoid_1:y:0'gru_6/while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/add_2AddV2%gru_6/while/gru_cell_6/split:output:2gru_6/while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_6/while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/while/gru_cell_6/mul_1Mul$gru_6/while/gru_cell_6/beta:output:0 gru_6/while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_6/while/gru_cell_6/Sigmoid_2Sigmoid gru_6/while/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/mul_2Mul gru_6/while/gru_cell_6/add_2:z:0$gru_6/while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/IdentityIdentity gru_6/while/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdç
 gru_6/while/gru_cell_6/IdentityN	IdentityN gru_6/while/gru_cell_6/mul_2:z:0 gru_6/while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48267*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/mul_3Mul"gru_6/while/gru_cell_6/Sigmoid:y:0gru_6_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_6/while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/while/gru_cell_6/subSub%gru_6/while/gru_cell_6/sub/x:output:0"gru_6/while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_6/while/gru_cell_6/mul_4Mulgru_6/while/gru_cell_6/sub:z:0)gru_6/while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/while/gru_cell_6/add_3AddV2 gru_6/while/gru_cell_6/mul_3:z:0 gru_6/while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_6_while_placeholder_1gru_6_while_placeholder gru_6/while/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_6/while/addAddV2gru_6_while_placeholdergru_6/while/add/y:output:0*
T0*
_output_shapes
: U
gru_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_6/while/add_1AddV2$gru_6_while_gru_6_while_loop_countergru_6/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_6/while/IdentityIdentitygru_6/while/add_1:z:0^gru_6/while/NoOp*
T0*
_output_shapes
: 
gru_6/while/Identity_1Identity*gru_6_while_gru_6_while_maximum_iterations^gru_6/while/NoOp*
T0*
_output_shapes
: k
gru_6/while/Identity_2Identitygru_6/while/add:z:0^gru_6/while/NoOp*
T0*
_output_shapes
: «
gru_6/while/Identity_3Identity@gru_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_6/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_6/while/Identity_4Identity gru_6/while/gru_cell_6/add_3:z:0^gru_6/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_6/while/NoOpNoOp-^gru_6/while/gru_cell_6/MatMul/ReadVariableOp/^gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp&^gru_6/while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_6_while_gru_6_strided_slice_1#gru_6_while_gru_6_strided_slice_1_0"t
7gru_6_while_gru_cell_6_matmul_1_readvariableop_resource9gru_6_while_gru_cell_6_matmul_1_readvariableop_resource_0"p
5gru_6_while_gru_cell_6_matmul_readvariableop_resource7gru_6_while_gru_cell_6_matmul_readvariableop_resource_0"b
.gru_6_while_gru_cell_6_readvariableop_resource0gru_6_while_gru_cell_6_readvariableop_resource_0"5
gru_6_while_identitygru_6/while/Identity:output:0"9
gru_6_while_identity_1gru_6/while/Identity_1:output:0"9
gru_6_while_identity_2gru_6/while/Identity_2:output:0"9
gru_6_while_identity_3gru_6/while/Identity_3:output:0"9
gru_6_while_identity_4gru_6/while/Identity_4:output:0"À
]gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_gru_6_while_tensorarrayv2read_tensorlistgetitem_gru_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_6/while/gru_cell_6/MatMul/ReadVariableOp,gru_6/while/gru_cell_6/MatMul/ReadVariableOp2`
.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp.gru_6/while/gru_cell_6/MatMul_1/ReadVariableOp2N
%gru_6/while/gru_cell_6/ReadVariableOp%gru_6/while/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
êU
õ
__inference__traced_save_53019
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_gru_6_gru_cell_6_kernel_read_readvariableop@
<savev2_gru_6_gru_cell_6_recurrent_kernel_read_readvariableop4
0savev2_gru_6_gru_cell_6_bias_read_readvariableop6
2savev2_gru_7_gru_cell_7_kernel_read_readvariableop@
<savev2_gru_7_gru_cell_7_recurrent_kernel_read_readvariableop4
0savev2_gru_7_gru_cell_7_bias_read_readvariableop6
2savev2_gru_8_gru_cell_8_kernel_read_readvariableop@
<savev2_gru_8_gru_cell_8_recurrent_kernel_read_readvariableop4
0savev2_gru_8_gru_cell_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop=
9savev2_adam_gru_6_gru_cell_6_kernel_m_read_readvariableopG
Csavev2_adam_gru_6_gru_cell_6_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_6_gru_cell_6_bias_m_read_readvariableop=
9savev2_adam_gru_7_gru_cell_7_kernel_m_read_readvariableopG
Csavev2_adam_gru_7_gru_cell_7_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_7_gru_cell_7_bias_m_read_readvariableop=
9savev2_adam_gru_8_gru_cell_8_kernel_m_read_readvariableopG
Csavev2_adam_gru_8_gru_cell_8_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_8_gru_cell_8_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop=
9savev2_adam_gru_6_gru_cell_6_kernel_v_read_readvariableopG
Csavev2_adam_gru_6_gru_cell_6_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_6_gru_cell_6_bias_v_read_readvariableop=
9savev2_adam_gru_7_gru_cell_7_kernel_v_read_readvariableopG
Csavev2_adam_gru_7_gru_cell_7_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_7_gru_cell_7_bias_v_read_readvariableop=
9savev2_adam_gru_8_gru_cell_8_kernel_v_read_readvariableopG
Csavev2_adam_gru_8_gru_cell_8_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_8_gru_cell_8_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Å
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*î
valueäBá)B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¿
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Â
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_gru_6_gru_cell_6_kernel_read_readvariableop<savev2_gru_6_gru_cell_6_recurrent_kernel_read_readvariableop0savev2_gru_6_gru_cell_6_bias_read_readvariableop2savev2_gru_7_gru_cell_7_kernel_read_readvariableop<savev2_gru_7_gru_cell_7_recurrent_kernel_read_readvariableop0savev2_gru_7_gru_cell_7_bias_read_readvariableop2savev2_gru_8_gru_cell_8_kernel_read_readvariableop<savev2_gru_8_gru_cell_8_recurrent_kernel_read_readvariableop0savev2_gru_8_gru_cell_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop9savev2_adam_gru_6_gru_cell_6_kernel_m_read_readvariableopCsavev2_adam_gru_6_gru_cell_6_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_6_gru_cell_6_bias_m_read_readvariableop9savev2_adam_gru_7_gru_cell_7_kernel_m_read_readvariableopCsavev2_adam_gru_7_gru_cell_7_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_7_gru_cell_7_bias_m_read_readvariableop9savev2_adam_gru_8_gru_cell_8_kernel_m_read_readvariableopCsavev2_adam_gru_8_gru_cell_8_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_8_gru_cell_8_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop9savev2_adam_gru_6_gru_cell_6_kernel_v_read_readvariableopCsavev2_adam_gru_6_gru_cell_6_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_6_gru_cell_6_bias_v_read_readvariableop9savev2_adam_gru_7_gru_cell_7_kernel_v_read_readvariableopCsavev2_adam_gru_7_gru_cell_7_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_7_gru_cell_7_bias_v_read_readvariableop9savev2_adam_gru_8_gru_cell_8_kernel_v_read_readvariableopCsavev2_adam_gru_8_gru_cell_8_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_8_gru_cell_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesî
ë: :d:: : : : : :	¬:	d¬:	¬:	d¬:	d¬:	¬:	d¬:	d¬:	¬: : :d::	¬:	d¬:	¬:	d¬:	d¬:	¬:	d¬:	d¬:	¬:d::	¬:	d¬:	¬:	d¬:	d¬:	¬:	d¬:	d¬:	¬: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	¬:%	!

_output_shapes
:	d¬:%
!

_output_shapes
:	¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	d¬:%!

_output_shapes
:	¬:$ 

_output_shapes

:d: 

_output_shapes
::% !

_output_shapes
:	¬:%!!

_output_shapes
:	d¬:%"!

_output_shapes
:	¬:%#!

_output_shapes
:	d¬:%$!

_output_shapes
:	d¬:%%!

_output_shapes
:	¬:%&!

_output_shapes
:	d¬:%'!

_output_shapes
:	d¬:%(!

_output_shapes
:	¬:)

_output_shapes
: 
Õ
¥
while_cond_47191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47191___redundant_placeholder03
/while_while_cond_47191___redundant_placeholder13
/while_while_cond_47191___redundant_placeholder23
/while_while_cond_47191___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Õ
¥
while_cond_50333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_50333___redundant_placeholder03
/while_while_cond_50333___redundant_placeholder13
/while_while_cond_50333___redundant_placeholder23
/while_while_cond_50333___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
¢
¥
"__inference_internal_grad_fn_52312
result_grads_0
result_grads_1#
mul_gru_8_while_gru_cell_8_beta$
 mul_gru_8_while_gru_cell_8_add_2
identity
mulMulmul_gru_8_while_gru_cell_8_beta mul_gru_8_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_8_while_gru_cell_8_beta mul_gru_8_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý

"__inference_internal_grad_fn_52258
result_grads_0
result_grads_1
mul_gru_8_gru_cell_8_beta
mul_gru_8_gru_cell_8_add_2
identity
mulMulmul_gru_8_gru_cell_8_betamul_gru_8_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_8_gru_cell_8_betamul_gru_8_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
å

§
,__inference_sequential_2_layer_call_fn_48146

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
	unknown_2:	¬
	unknown_3:	d¬
	unknown_4:	d¬
	unknown_5:	¬
	unknown_6:	d¬
	unknown_7:	d¬
	unknown_8:d
	unknown_9:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_47974o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
B
þ
while_body_49789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_6_readvariableop_resource_0:	¬D
1while_gru_cell_6_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_6_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_6_readvariableop_resource:	¬B
/while_gru_cell_6_matmul_readvariableop_resource:	¬D
1while_gru_cell_6_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_6/MatMul/ReadVariableOp¢(while/gru_cell_6/MatMul_1/ReadVariableOp¢while/gru_cell_6/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_6/ReadVariableOpReadVariableOp*while_gru_cell_6_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_6/unstackUnpack'while/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_6/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_6/BiasAddBiasAdd!while/gru_cell_6/MatMul:product:0!while/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_6/splitSplit)while/gru_cell_6/split/split_dim:output:0!while/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_6/MatMul_1MatMulwhile_placeholder_20while/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_6/BiasAdd_1BiasAdd#while/gru_cell_6/MatMul_1:product:0!while/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_6/split_1SplitV#while/gru_cell_6/BiasAdd_1:output:0while/gru_cell_6/Const:output:0+while/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_6/addAddV2while/gru_cell_6/split:output:0!while/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_6/SigmoidSigmoidwhile/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_1AddV2while/gru_cell_6/split:output:1!while/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_1Sigmoidwhile/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mulMulwhile/gru_cell_6/Sigmoid_1:y:0!while/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_2AddV2while/gru_cell_6/split:output:2while/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/mul_1Mulwhile/gru_cell_6/beta:output:0while/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/Sigmoid_2Sigmoidwhile/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_2Mulwhile/gru_cell_6/add_2:z:0while/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_6/IdentityIdentitywhile/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_6/IdentityN	IdentityNwhile/gru_cell_6/mul_2:z:0while/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-49839*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_3Mulwhile/gru_cell_6/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_6/subSubwhile/gru_cell_6/sub/x:output:0while/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/mul_4Mulwhile/gru_cell_6/sub:z:0#while/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_6/add_3AddV2while/gru_cell_6/mul_3:z:0while/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_6/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_6/MatMul/ReadVariableOp)^while/gru_cell_6/MatMul_1/ReadVariableOp ^while/gru_cell_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_6_matmul_1_readvariableop_resource3while_gru_cell_6_matmul_1_readvariableop_resource_0"d
/while_gru_cell_6_matmul_readvariableop_resource1while_gru_cell_6_matmul_readvariableop_resource_0"V
(while_gru_cell_6_readvariableop_resource*while_gru_cell_6_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_6/MatMul/ReadVariableOp&while/gru_cell_6/MatMul/ReadVariableOp2T
(while/gru_cell_6/MatMul_1/ReadVariableOp(while/gru_cell_6/MatMul_1/ReadVariableOp2B
while/gru_cell_6/ReadVariableOpwhile/gru_cell_6/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
õ
¶
%__inference_gru_8_layer_call_fn_50630

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_47288o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
B
þ
while_body_47192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_8_readvariableop_resource_0:	¬D
1while_gru_cell_8_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_8_readvariableop_resource:	¬B
/while_gru_cell_8_matmul_readvariableop_resource:	d¬D
1while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_8/MatMul/ReadVariableOp¢(while/gru_cell_8/MatMul_1/ReadVariableOp¢while/gru_cell_8/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_8/ReadVariableOpReadVariableOp*while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_8/unstackUnpack'while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_8/BiasAddBiasAdd!while/gru_cell_8/MatMul:product:0!while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_8/splitSplit)while/gru_cell_8/split/split_dim:output:0!while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_8/MatMul_1MatMulwhile_placeholder_20while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_8/BiasAdd_1BiasAdd#while/gru_cell_8/MatMul_1:product:0!while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_8/split_1SplitV#while/gru_cell_8/BiasAdd_1:output:0while/gru_cell_8/Const:output:0+while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_8/addAddV2while/gru_cell_8/split:output:0!while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_8/SigmoidSigmoidwhile/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_1AddV2while/gru_cell_8/split:output:1!while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_1Sigmoidwhile/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mulMulwhile/gru_cell_8/Sigmoid_1:y:0!while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_2AddV2while/gru_cell_8/split:output:2while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/mul_1Mulwhile/gru_cell_8/beta:output:0while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_2Sigmoidwhile/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_2Mulwhile/gru_cell_8/add_2:z:0while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/IdentityIdentitywhile/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_8/IdentityN	IdentityNwhile/gru_cell_8/mul_2:z:0while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47242*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_3Mulwhile/gru_cell_8/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/subSubwhile/gru_cell_8/sub/x:output:0while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_4Mulwhile/gru_cell_8/sub:z:0#while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_3AddV2while/gru_cell_8/mul_3:z:0while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_8/MatMul/ReadVariableOp)^while/gru_cell_8/MatMul_1/ReadVariableOp ^while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_8_matmul_1_readvariableop_resource3while_gru_cell_8_matmul_1_readvariableop_resource_0"d
/while_gru_cell_8_matmul_readvariableop_resource1while_gru_cell_8_matmul_readvariableop_resource_0"V
(while_gru_cell_8_readvariableop_resource*while_gru_cell_8_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_8/MatMul/ReadVariableOp&while/gru_cell_8/MatMul/ReadVariableOp2T
(while/gru_cell_8/MatMul_1/ReadVariableOp(while/gru_cell_8/MatMul_1/ReadVariableOp2B
while/gru_cell_8/ReadVariableOpwhile/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
!
Û
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51642

inputs
states_0*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51628*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdU
mul_3MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
÷

gru_7_while_cond_48379(
$gru_7_while_gru_7_while_loop_counter.
*gru_7_while_gru_7_while_maximum_iterations
gru_7_while_placeholder
gru_7_while_placeholder_1
gru_7_while_placeholder_2*
&gru_7_while_less_gru_7_strided_slice_1?
;gru_7_while_gru_7_while_cond_48379___redundant_placeholder0?
;gru_7_while_gru_7_while_cond_48379___redundant_placeholder1?
;gru_7_while_gru_7_while_cond_48379___redundant_placeholder2?
;gru_7_while_gru_7_while_cond_48379___redundant_placeholder3
gru_7_while_identity
z
gru_7/while/LessLessgru_7_while_placeholder&gru_7_while_less_gru_7_strided_slice_1*
T0*
_output_shapes
: W
gru_7/while/IdentityIdentitygru_7/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_7_while_identitygru_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ý
¶
%__inference_gru_6_layer_call_fn_49217

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_47906s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ý

"__inference_internal_grad_fn_52060
result_grads_0
result_grads_1
mul_while_gru_cell_7_beta
mul_while_gru_cell_7_add_2
identity
mulMulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_50501
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_7_readvariableop_resource_0:	¬D
1while_gru_cell_7_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_7_readvariableop_resource:	¬B
/while_gru_cell_7_matmul_readvariableop_resource:	d¬D
1while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_7/MatMul/ReadVariableOp¢(while/gru_cell_7/MatMul_1/ReadVariableOp¢while/gru_cell_7/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_7/ReadVariableOpReadVariableOp*while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_7/unstackUnpack'while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_7/BiasAddBiasAdd!while/gru_cell_7/MatMul:product:0!while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_7/splitSplit)while/gru_cell_7/split/split_dim:output:0!while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_7/MatMul_1MatMulwhile_placeholder_20while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_7/BiasAdd_1BiasAdd#while/gru_cell_7/MatMul_1:product:0!while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_7/split_1SplitV#while/gru_cell_7/BiasAdd_1:output:0while/gru_cell_7/Const:output:0+while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_7/addAddV2while/gru_cell_7/split:output:0!while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_7/SigmoidSigmoidwhile/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_1AddV2while/gru_cell_7/split:output:1!while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_1Sigmoidwhile/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mulMulwhile/gru_cell_7/Sigmoid_1:y:0!while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_2AddV2while/gru_cell_7/split:output:2while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/mul_1Mulwhile/gru_cell_7/beta:output:0while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_2Sigmoidwhile/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_2Mulwhile/gru_cell_7/add_2:z:0while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/IdentityIdentitywhile/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_7/IdentityN	IdentityNwhile/gru_cell_7/mul_2:z:0while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50551*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_3Mulwhile/gru_cell_7/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/subSubwhile/gru_cell_7/sub/x:output:0while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_4Mulwhile/gru_cell_7/sub:z:0#while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_3AddV2while/gru_cell_7/mul_3:z:0while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_7/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_7/MatMul/ReadVariableOp)^while/gru_cell_7/MatMul_1/ReadVariableOp ^while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_7_matmul_1_readvariableop_resource3while_gru_cell_7_matmul_1_readvariableop_resource_0"d
/while_gru_cell_7_matmul_readvariableop_resource1while_gru_cell_7_matmul_readvariableop_resource_0"V
(while_gru_cell_7_readvariableop_resource*while_gru_cell_7_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_7/MatMul/ReadVariableOp&while/gru_cell_7/MatMul/ReadVariableOp2T
(while/gru_cell_7/MatMul_1/ReadVariableOp(while/gru_cell_7/MatMul_1/ReadVariableOp2B
while/gru_cell_7/ReadVariableOpwhile/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
÷

gru_8_while_cond_49041(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2*
&gru_8_while_less_gru_8_strided_slice_1?
;gru_8_while_gru_8_while_cond_49041___redundant_placeholder0?
;gru_8_while_gru_8_while_cond_49041___redundant_placeholder1?
;gru_8_while_gru_8_while_cond_49041___redundant_placeholder2?
;gru_8_while_gru_8_while_cond_49041___redundant_placeholder3
gru_8_while_identity
z
gru_8/while/LessLessgru_8_while_placeholder&gru_8_while_less_gru_8_strided_slice_1*
T0*
_output_shapes
: W
gru_8/while/IdentityIdentitygru_8/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_8_while_identitygru_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
¢
¥
"__inference_internal_grad_fn_52168
result_grads_0
result_grads_1#
mul_gru_6_while_gru_cell_6_beta$
 mul_gru_6_while_gru_cell_6_add_2
identity
mulMulmul_gru_6_while_gru_cell_6_beta mul_gru_6_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_6_while_gru_cell_6_beta mul_gru_6_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
Û
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51522

inputs
states_0*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51508*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdU
mul_3MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
Ì
³
"__inference_internal_grad_fn_51808
result_grads_0
result_grads_1*
&mul_sequential_2_gru_7_gru_cell_7_beta+
'mul_sequential_2_gru_7_gru_cell_7_add_2
identity
mulMul&mul_sequential_2_gru_7_gru_cell_7_beta'mul_sequential_2_gru_7_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul&mul_sequential_2_gru_7_gru_cell_7_beta'mul_sequential_2_gru_7_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
­
Ë
G__inference_sequential_2_layer_call_and_return_conditional_losses_48086
gru_6_input
gru_6_48059:	¬
gru_6_48061:	¬
gru_6_48063:	d¬
gru_7_48066:	¬
gru_7_48068:	d¬
gru_7_48070:	d¬
gru_8_48073:	¬
gru_8_48075:	d¬
gru_8_48077:	d¬
dense_2_48080:d
dense_2_48082:
identity¢dense_2/StatefulPartitionedCall¢gru_6/StatefulPartitionedCall¢gru_7/StatefulPartitionedCall¢gru_8/StatefulPartitionedCallù
gru_6/StatefulPartitionedCallStatefulPartitionedCallgru_6_inputgru_6_48059gru_6_48061gru_6_48063*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_47906
gru_7/StatefulPartitionedCallStatefulPartitionedCall&gru_6/StatefulPartitionedCall:output:0gru_7_48066gru_7_48068gru_7_48070*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_47717
gru_8/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0gru_8_48073gru_8_48075gru_8_48077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_47528
dense_2/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0dense_2_48080dense_2_48082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_47306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_2/StatefulPartitionedCall^gru_6/StatefulPartitionedCall^gru_7/StatefulPartitionedCall^gru_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
gru_6/StatefulPartitionedCallgru_6/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_6_input
Ø

"__inference_internal_grad_fn_51898
result_grads_0
result_grads_1
mul_gru_cell_6_beta
mul_gru_cell_6_add_2
identityx
mulMulmul_gru_cell_6_betamul_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_6_betamul_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_47018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_7_readvariableop_resource_0:	¬D
1while_gru_cell_7_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_7_readvariableop_resource:	¬B
/while_gru_cell_7_matmul_readvariableop_resource:	d¬D
1while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_7/MatMul/ReadVariableOp¢(while/gru_cell_7/MatMul_1/ReadVariableOp¢while/gru_cell_7/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_7/ReadVariableOpReadVariableOp*while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_7/unstackUnpack'while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_7/BiasAddBiasAdd!while/gru_cell_7/MatMul:product:0!while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_7/splitSplit)while/gru_cell_7/split/split_dim:output:0!while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_7/MatMul_1MatMulwhile_placeholder_20while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_7/BiasAdd_1BiasAdd#while/gru_cell_7/MatMul_1:product:0!while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_7/split_1SplitV#while/gru_cell_7/BiasAdd_1:output:0while/gru_cell_7/Const:output:0+while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_7/addAddV2while/gru_cell_7/split:output:0!while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_7/SigmoidSigmoidwhile/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_1AddV2while/gru_cell_7/split:output:1!while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_1Sigmoidwhile/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mulMulwhile/gru_cell_7/Sigmoid_1:y:0!while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_2AddV2while/gru_cell_7/split:output:2while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/mul_1Mulwhile/gru_cell_7/beta:output:0while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_2Sigmoidwhile/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_2Mulwhile/gru_cell_7/add_2:z:0while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/IdentityIdentitywhile/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_7/IdentityN	IdentityNwhile/gru_cell_7/mul_2:z:0while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47068*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_3Mulwhile/gru_cell_7/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/subSubwhile/gru_cell_7/sub/x:output:0while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_4Mulwhile/gru_cell_7/sub:z:0#while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_3AddV2while/gru_cell_7/mul_3:z:0while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_7/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_7/MatMul/ReadVariableOp)^while/gru_cell_7/MatMul_1/ReadVariableOp ^while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_7_matmul_1_readvariableop_resource3while_gru_cell_7_matmul_1_readvariableop_resource_0"d
/while_gru_cell_7_matmul_readvariableop_resource1while_gru_cell_7_matmul_readvariableop_resource_0"V
(while_gru_cell_7_readvariableop_resource*while_gru_cell_7_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_7/MatMul/ReadVariableOp&while/gru_cell_7/MatMul/ReadVariableOp2T
(while/gru_cell_7/MatMul_1/ReadVariableOp(while/gru_cell_7/MatMul_1/ReadVariableOp2B
while/gru_cell_7/ReadVariableOpwhile/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý

"__inference_internal_grad_fn_52600
result_grads_0
result_grads_1
mul_while_gru_cell_7_beta
mul_while_gru_cell_7_add_2
identity
mulMulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¢
¥
"__inference_internal_grad_fn_52204
result_grads_0
result_grads_1#
mul_gru_8_while_gru_cell_8_beta$
 mul_gru_8_while_gru_cell_8_add_2
identity
mulMulmul_gru_8_while_gru_cell_8_beta mul_gru_8_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_8_while_gru_cell_8_beta mul_gru_8_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
B
þ
while_body_50334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_7_readvariableop_resource_0:	¬D
1while_gru_cell_7_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_7_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_7_readvariableop_resource:	¬B
/while_gru_cell_7_matmul_readvariableop_resource:	d¬D
1while_gru_cell_7_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_7/MatMul/ReadVariableOp¢(while/gru_cell_7/MatMul_1/ReadVariableOp¢while/gru_cell_7/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_7/ReadVariableOpReadVariableOp*while_gru_cell_7_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_7/unstackUnpack'while/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_7/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_7_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_7/BiasAddBiasAdd!while/gru_cell_7/MatMul:product:0!while/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_7/splitSplit)while/gru_cell_7/split/split_dim:output:0!while/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_7/MatMul_1MatMulwhile_placeholder_20while/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_7/BiasAdd_1BiasAdd#while/gru_cell_7/MatMul_1:product:0!while/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_7/split_1SplitV#while/gru_cell_7/BiasAdd_1:output:0while/gru_cell_7/Const:output:0+while/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_7/addAddV2while/gru_cell_7/split:output:0!while/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_7/SigmoidSigmoidwhile/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_1AddV2while/gru_cell_7/split:output:1!while/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_1Sigmoidwhile/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mulMulwhile/gru_cell_7/Sigmoid_1:y:0!while/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_2AddV2while/gru_cell_7/split:output:2while/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/mul_1Mulwhile/gru_cell_7/beta:output:0while/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/Sigmoid_2Sigmoidwhile/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_2Mulwhile/gru_cell_7/add_2:z:0while/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_7/IdentityIdentitywhile/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_7/IdentityN	IdentityNwhile/gru_cell_7/mul_2:z:0while/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50384*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_3Mulwhile/gru_cell_7/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_7/subSubwhile/gru_cell_7/sub/x:output:0while/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/mul_4Mulwhile/gru_cell_7/sub:z:0#while/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_7/add_3AddV2while/gru_cell_7/mul_3:z:0while/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_7/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_7/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_7/MatMul/ReadVariableOp)^while/gru_cell_7/MatMul_1/ReadVariableOp ^while/gru_cell_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_7_matmul_1_readvariableop_resource3while_gru_cell_7_matmul_1_readvariableop_resource_0"d
/while_gru_cell_7_matmul_readvariableop_resource1while_gru_cell_7_matmul_readvariableop_resource_0"V
(while_gru_cell_7_readvariableop_resource*while_gru_cell_7_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_7/MatMul/ReadVariableOp&while/gru_cell_7/MatMul/ReadVariableOp2T
(while/gru_cell_7/MatMul_1/ReadVariableOp(while/gru_cell_7/MatMul_1/ReadVariableOp2B
while/gru_cell_7/ReadVariableOpwhile/gru_cell_7/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý

"__inference_internal_grad_fn_52384
result_grads_0
result_grads_1
mul_while_gru_cell_6_beta
mul_while_gru_cell_6_add_2
identity
mulMulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
å

§
,__inference_sequential_2_layer_call_fn_48119

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
	unknown_2:	¬
	unknown_3:	d¬
	unknown_4:	d¬
	unknown_5:	¬
	unknown_6:	d¬
	unknown_7:	d¬
	unknown_8:d
	unknown_9:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_47313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
!
Û
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51688

inputs
states_0*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51674*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdU
mul_3MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0

w
"__inference_internal_grad_fn_52510
result_grads_0
result_grads_1
mul_beta
	mul_add_2
identityb
mulMulmul_beta	mul_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
mul_1Mulmul_beta	mul_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý
¶
%__inference_gru_7_layer_call_fn_49918

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_47114s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
­
Ë
G__inference_sequential_2_layer_call_and_return_conditional_losses_48056
gru_6_input
gru_6_48029:	¬
gru_6_48031:	¬
gru_6_48033:	d¬
gru_7_48036:	¬
gru_7_48038:	d¬
gru_7_48040:	d¬
gru_8_48043:	¬
gru_8_48045:	d¬
gru_8_48047:	d¬
dense_2_48050:d
dense_2_48052:
identity¢dense_2/StatefulPartitionedCall¢gru_6/StatefulPartitionedCall¢gru_7/StatefulPartitionedCall¢gru_8/StatefulPartitionedCallù
gru_6/StatefulPartitionedCallStatefulPartitionedCallgru_6_inputgru_6_48029gru_6_48031gru_6_48033*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_6_layer_call_and_return_conditional_losses_46940
gru_7/StatefulPartitionedCallStatefulPartitionedCall&gru_6/StatefulPartitionedCall:output:0gru_7_48036gru_7_48038gru_7_48040*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_7_layer_call_and_return_conditional_losses_47114
gru_8/StatefulPartitionedCallStatefulPartitionedCall&gru_7/StatefulPartitionedCall:output:0gru_8_48043gru_8_48045gru_8_48047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_gru_8_layer_call_and_return_conditional_losses_47288
dense_2/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0dense_2_48050dense_2_48052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_47306w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_2/StatefulPartitionedCall^gru_6/StatefulPartitionedCall^gru_7/StatefulPartitionedCall^gru_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
gru_6/StatefulPartitionedCallgru_6/StatefulPartitionedCall2>
gru_7/StatefulPartitionedCallgru_7/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_6_input
!
Ù
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46140

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	d¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-46126*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdS
mul_3MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
B
þ
while_body_51213
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_8_readvariableop_resource_0:	¬D
1while_gru_cell_8_matmul_readvariableop_resource_0:	d¬F
3while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_8_readvariableop_resource:	¬B
/while_gru_cell_8_matmul_readvariableop_resource:	d¬D
1while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_8/MatMul/ReadVariableOp¢(while/gru_cell_8/MatMul_1/ReadVariableOp¢while/gru_cell_8/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
while/gru_cell_8/ReadVariableOpReadVariableOp*while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_8/unstackUnpack'while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¶
while/gru_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_8/BiasAddBiasAdd!while/gru_cell_8/MatMul:product:0!while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_8/splitSplit)while/gru_cell_8/split/split_dim:output:0!while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_8/MatMul_1MatMulwhile_placeholder_20while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_8/BiasAdd_1BiasAdd#while/gru_cell_8/MatMul_1:product:0!while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_8/split_1SplitV#while/gru_cell_8/BiasAdd_1:output:0while/gru_cell_8/Const:output:0+while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_8/addAddV2while/gru_cell_8/split:output:0!while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_8/SigmoidSigmoidwhile/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_1AddV2while/gru_cell_8/split:output:1!while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_1Sigmoidwhile/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mulMulwhile/gru_cell_8/Sigmoid_1:y:0!while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_2AddV2while/gru_cell_8/split:output:2while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/mul_1Mulwhile/gru_cell_8/beta:output:0while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/Sigmoid_2Sigmoidwhile/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_2Mulwhile/gru_cell_8/add_2:z:0while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_8/IdentityIdentitywhile/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
while/gru_cell_8/IdentityN	IdentityNwhile/gru_cell_8/mul_2:z:0while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51263*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_3Mulwhile/gru_cell_8/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_8/subSubwhile/gru_cell_8/sub/x:output:0while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/mul_4Mulwhile/gru_cell_8/sub:z:0#while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_8/add_3AddV2while/gru_cell_8/mul_3:z:0while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_8/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_8/MatMul/ReadVariableOp)^while/gru_cell_8/MatMul_1/ReadVariableOp ^while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_8_matmul_1_readvariableop_resource3while_gru_cell_8_matmul_1_readvariableop_resource_0"d
/while_gru_cell_8_matmul_readvariableop_resource1while_gru_cell_8_matmul_readvariableop_resource_0"V
(while_gru_cell_8_readvariableop_resource*while_gru_cell_8_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2P
&while/gru_cell_8/MatMul/ReadVariableOp&while/gru_cell_8/MatMul/ReadVariableOp2T
(while/gru_cell_8/MatMul_1/ReadVariableOp(while/gru_cell_8/MatMul_1/ReadVariableOp2B
while/gru_cell_8/ReadVariableOpwhile/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Å	
ó
B__inference_dense_2_layer_call_and_return_conditional_losses_47306

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
å
§
while_body_45990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_6_46012_0:	¬+
while_gru_cell_6_46014_0:	¬+
while_gru_cell_6_46016_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_6_46012:	¬)
while_gru_cell_6_46014:	¬)
while_gru_cell_6_46016:	d¬¢(while/gru_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_6_46012_0while_gru_cell_6_46014_0while_gru_cell_6_46016_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45938Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_6_46012while_gru_cell_6_46012_0"2
while_gru_cell_6_46014while_gru_cell_6_46014_0"2
while_gru_cell_6_46016while_gru_cell_6_46016_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/gru_cell_6/StatefulPartitionedCall(while/gru_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
°

Ù
*__inference_gru_cell_7_layer_call_fn_51462

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46140o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
Õ
¥
while_cond_50166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_50166___redundant_placeholder03
/while_while_cond_50166___redundant_placeholder13
/while_while_cond_50166___redundant_placeholder23
/while_while_cond_50166___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ÆQ

@__inference_gru_7_layer_call_and_return_conditional_losses_47114

inputs5
"gru_cell_7_readvariableop_resource:	¬<
)gru_cell_7_matmul_readvariableop_resource:	d¬>
+gru_cell_7_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_7/MatMul/ReadVariableOp¢"gru_cell_7/MatMul_1/ReadVariableOp¢gru_cell_7/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_7/ReadVariableOpReadVariableOp"gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_7/unstackUnpack!gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_7/MatMul/ReadVariableOpReadVariableOp)gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMulMatMulstrided_slice_2:output:0(gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAddBiasAddgru_cell_7/MatMul:product:0gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_7/splitSplit#gru_cell_7/split/split_dim:output:0gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMul_1MatMulzeros:output:0*gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAdd_1BiasAddgru_cell_7/MatMul_1:product:0gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_7/split_1SplitVgru_cell_7/BiasAdd_1:output:0gru_cell_7/Const:output:0%gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_7/addAddV2gru_cell_7/split:output:0gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_7/SigmoidSigmoidgru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_7/add_1AddV2gru_cell_7/split:output:1gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_1Sigmoidgru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_7/mulMulgru_cell_7/Sigmoid_1:y:0gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_7/add_2AddV2gru_cell_7/split:output:2gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_7/mul_1Mulgru_cell_7/beta:output:0gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_2Sigmoidgru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_7/mul_2Mulgru_cell_7/add_2:z:0gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/IdentityIdentitygru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_7/IdentityN	IdentityNgru_cell_7/mul_2:z:0gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-47002*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_7/mul_3Mulgru_cell_7/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_7/subSubgru_cell_7/sub/x:output:0gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_7/mul_4Mulgru_cell_7/sub:z:0gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_7/add_3AddV2gru_cell_7/mul_3:z:0gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_7_readvariableop_resource)gru_cell_7_matmul_readvariableop_resource+gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_47018*
condR
while_cond_47017*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_7/MatMul/ReadVariableOp#^gru_cell_7/MatMul_1/ReadVariableOp^gru_cell_7/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_7/MatMul/ReadVariableOp gru_cell_7/MatMul/ReadVariableOp2H
"gru_cell_7/MatMul_1/ReadVariableOp"gru_cell_7/MatMul_1/ReadVariableOp26
gru_cell_7/ReadVariableOpgru_cell_7/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs

ù	
G__inference_sequential_2_layer_call_and_return_conditional_losses_48645

inputs;
(gru_6_gru_cell_6_readvariableop_resource:	¬B
/gru_6_gru_cell_6_matmul_readvariableop_resource:	¬D
1gru_6_gru_cell_6_matmul_1_readvariableop_resource:	d¬;
(gru_7_gru_cell_7_readvariableop_resource:	¬B
/gru_7_gru_cell_7_matmul_readvariableop_resource:	d¬D
1gru_7_gru_cell_7_matmul_1_readvariableop_resource:	d¬;
(gru_8_gru_cell_8_readvariableop_resource:	¬B
/gru_8_gru_cell_8_matmul_readvariableop_resource:	d¬D
1gru_8_gru_cell_8_matmul_1_readvariableop_resource:	d¬8
&dense_2_matmul_readvariableop_resource:d5
'dense_2_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢&gru_6/gru_cell_6/MatMul/ReadVariableOp¢(gru_6/gru_cell_6/MatMul_1/ReadVariableOp¢gru_6/gru_cell_6/ReadVariableOp¢gru_6/while¢&gru_7/gru_cell_7/MatMul/ReadVariableOp¢(gru_7/gru_cell_7/MatMul_1/ReadVariableOp¢gru_7/gru_cell_7/ReadVariableOp¢gru_7/while¢&gru_8/gru_cell_8/MatMul/ReadVariableOp¢(gru_8/gru_cell_8/MatMul_1/ReadVariableOp¢gru_8/gru_cell_8/ReadVariableOp¢gru_8/whileA
gru_6/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_6/strided_sliceStridedSlicegru_6/Shape:output:0"gru_6/strided_slice/stack:output:0$gru_6/strided_slice/stack_1:output:0$gru_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_6/zeros/packedPackgru_6/strided_slice:output:0gru_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_6/zerosFillgru_6/zeros/packed:output:0gru_6/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_6/transpose	Transposeinputsgru_6/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿP
gru_6/Shape_1Shapegru_6/transpose:y:0*
T0*
_output_shapes
:e
gru_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_6/strided_slice_1StridedSlicegru_6/Shape_1:output:0$gru_6/strided_slice_1/stack:output:0&gru_6/strided_slice_1/stack_1:output:0&gru_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_6/TensorArrayV2TensorListReserve*gru_6/TensorArrayV2/element_shape:output:0gru_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_6/transpose:y:0Dgru_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_6/strided_slice_2StridedSlicegru_6/transpose:y:0$gru_6/strided_slice_2/stack:output:0&gru_6/strided_slice_2/stack_1:output:0&gru_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_6/gru_cell_6/ReadVariableOpReadVariableOp(gru_6_gru_cell_6_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_6/gru_cell_6/unstackUnpack'gru_6/gru_cell_6/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_6/gru_cell_6/MatMul/ReadVariableOpReadVariableOp/gru_6_gru_cell_6_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0¤
gru_6/gru_cell_6/MatMulMatMulgru_6/strided_slice_2:output:0.gru_6/gru_cell_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_6/gru_cell_6/BiasAddBiasAdd!gru_6/gru_cell_6/MatMul:product:0!gru_6/gru_cell_6/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_6/gru_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_6/gru_cell_6/splitSplit)gru_6/gru_cell_6/split/split_dim:output:0!gru_6/gru_cell_6/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_6/gru_cell_6/MatMul_1/ReadVariableOpReadVariableOp1gru_6_gru_cell_6_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_6/gru_cell_6/MatMul_1MatMulgru_6/zeros:output:00gru_6/gru_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_6/gru_cell_6/BiasAdd_1BiasAdd#gru_6/gru_cell_6/MatMul_1:product:0!gru_6/gru_cell_6/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_6/gru_cell_6/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_6/gru_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_6/gru_cell_6/split_1SplitV#gru_6/gru_cell_6/BiasAdd_1:output:0gru_6/gru_cell_6/Const:output:0+gru_6/gru_cell_6/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_6/gru_cell_6/addAddV2gru_6/gru_cell_6/split:output:0!gru_6/gru_cell_6/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_6/gru_cell_6/SigmoidSigmoidgru_6/gru_cell_6/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/add_1AddV2gru_6/gru_cell_6/split:output:1!gru_6/gru_cell_6/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_6/gru_cell_6/Sigmoid_1Sigmoidgru_6/gru_cell_6/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mulMulgru_6/gru_cell_6/Sigmoid_1:y:0!gru_6/gru_cell_6/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/add_2AddV2gru_6/gru_cell_6/split:output:2gru_6/gru_cell_6/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_6/gru_cell_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/gru_cell_6/mul_1Mulgru_6/gru_cell_6/beta:output:0gru_6/gru_cell_6/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_6/gru_cell_6/Sigmoid_2Sigmoidgru_6/gru_cell_6/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mul_2Mulgru_6/gru_cell_6/add_2:z:0gru_6/gru_cell_6/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_6/gru_cell_6/IdentityIdentitygru_6/gru_cell_6/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
gru_6/gru_cell_6/IdentityN	IdentityNgru_6/gru_cell_6/mul_2:z:0gru_6/gru_cell_6/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48201*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mul_3Mulgru_6/gru_cell_6/Sigmoid:y:0gru_6/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_6/gru_cell_6/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_6/gru_cell_6/subSubgru_6/gru_cell_6/sub/x:output:0gru_6/gru_cell_6/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/mul_4Mulgru_6/gru_cell_6/sub:z:0#gru_6/gru_cell_6/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_6/gru_cell_6/add_3AddV2gru_6/gru_cell_6/mul_3:z:0gru_6/gru_cell_6/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_6/TensorArrayV2_1TensorListReserve,gru_6/TensorArrayV2_1/element_shape:output:0gru_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_6/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_6/whileWhile!gru_6/while/loop_counter:output:0'gru_6/while/maximum_iterations:output:0gru_6/time:output:0gru_6/TensorArrayV2_1:handle:0gru_6/zeros:output:0gru_6/strided_slice_1:output:0=gru_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_6_gru_cell_6_readvariableop_resource/gru_6_gru_cell_6_matmul_readvariableop_resource1gru_6_gru_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *"
bodyR
gru_6_while_body_48217*"
condR
gru_6_while_cond_48216*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_6/TensorArrayV2Stack/TensorListStackTensorListStackgru_6/while:output:3?gru_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_6/strided_slice_3StridedSlice1gru_6/TensorArrayV2Stack/TensorListStack:tensor:0$gru_6/strided_slice_3/stack:output:0&gru_6/strided_slice_3/stack_1:output:0&gru_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_6/transpose_1	Transpose1gru_6/TensorArrayV2Stack/TensorListStack:tensor:0gru_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_6/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_7/ShapeShapegru_6/transpose_1:y:0*
T0*
_output_shapes
:c
gru_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_7/strided_sliceStridedSlicegru_7/Shape:output:0"gru_7/strided_slice/stack:output:0$gru_7/strided_slice/stack_1:output:0$gru_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_7/zeros/packedPackgru_7/strided_slice:output:0gru_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_7/zerosFillgru_7/zeros/packed:output:0gru_7/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_7/transpose	Transposegru_6/transpose_1:y:0gru_7/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_7/Shape_1Shapegru_7/transpose:y:0*
T0*
_output_shapes
:e
gru_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_7/strided_slice_1StridedSlicegru_7/Shape_1:output:0$gru_7/strided_slice_1/stack:output:0&gru_7/strided_slice_1/stack_1:output:0&gru_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_7/TensorArrayV2TensorListReserve*gru_7/TensorArrayV2/element_shape:output:0gru_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_7/transpose:y:0Dgru_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_7/strided_slice_2StridedSlicegru_7/transpose:y:0$gru_7/strided_slice_2/stack:output:0&gru_7/strided_slice_2/stack_1:output:0&gru_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
gru_7/gru_cell_7/ReadVariableOpReadVariableOp(gru_7_gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_7/gru_cell_7/unstackUnpack'gru_7/gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_7/gru_cell_7/MatMul/ReadVariableOpReadVariableOp/gru_7_gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¤
gru_7/gru_cell_7/MatMulMatMulgru_7/strided_slice_2:output:0.gru_7/gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_7/gru_cell_7/BiasAddBiasAdd!gru_7/gru_cell_7/MatMul:product:0!gru_7/gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_7/gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_7/gru_cell_7/splitSplit)gru_7/gru_cell_7/split/split_dim:output:0!gru_7/gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_7/gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp1gru_7_gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_7/gru_cell_7/MatMul_1MatMulgru_7/zeros:output:00gru_7/gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_7/gru_cell_7/BiasAdd_1BiasAdd#gru_7/gru_cell_7/MatMul_1:product:0!gru_7/gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_7/gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_7/gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_7/gru_cell_7/split_1SplitV#gru_7/gru_cell_7/BiasAdd_1:output:0gru_7/gru_cell_7/Const:output:0+gru_7/gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_7/gru_cell_7/addAddV2gru_7/gru_cell_7/split:output:0!gru_7/gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_7/gru_cell_7/SigmoidSigmoidgru_7/gru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/add_1AddV2gru_7/gru_cell_7/split:output:1!gru_7/gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_7/gru_cell_7/Sigmoid_1Sigmoidgru_7/gru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mulMulgru_7/gru_cell_7/Sigmoid_1:y:0!gru_7/gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/add_2AddV2gru_7/gru_cell_7/split:output:2gru_7/gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_7/gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/gru_cell_7/mul_1Mulgru_7/gru_cell_7/beta:output:0gru_7/gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_7/gru_cell_7/Sigmoid_2Sigmoidgru_7/gru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mul_2Mulgru_7/gru_cell_7/add_2:z:0gru_7/gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_7/gru_cell_7/IdentityIdentitygru_7/gru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
gru_7/gru_cell_7/IdentityN	IdentityNgru_7/gru_cell_7/mul_2:z:0gru_7/gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48364*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mul_3Mulgru_7/gru_cell_7/Sigmoid:y:0gru_7/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_7/gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_7/gru_cell_7/subSubgru_7/gru_cell_7/sub/x:output:0gru_7/gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/mul_4Mulgru_7/gru_cell_7/sub:z:0#gru_7/gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_7/gru_cell_7/add_3AddV2gru_7/gru_cell_7/mul_3:z:0gru_7/gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_7/TensorArrayV2_1TensorListReserve,gru_7/TensorArrayV2_1/element_shape:output:0gru_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_7/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_7/whileWhile!gru_7/while/loop_counter:output:0'gru_7/while/maximum_iterations:output:0gru_7/time:output:0gru_7/TensorArrayV2_1:handle:0gru_7/zeros:output:0gru_7/strided_slice_1:output:0=gru_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_7_gru_cell_7_readvariableop_resource/gru_7_gru_cell_7_matmul_readvariableop_resource1gru_7_gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *"
bodyR
gru_7_while_body_48380*"
condR
gru_7_while_cond_48379*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_7/TensorArrayV2Stack/TensorListStackTensorListStackgru_7/while:output:3?gru_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_7/strided_slice_3StridedSlice1gru_7/TensorArrayV2Stack/TensorListStack:tensor:0$gru_7/strided_slice_3/stack:output:0&gru_7/strided_slice_3/stack_1:output:0&gru_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_7/transpose_1	Transpose1gru_7/TensorArrayV2Stack/TensorListStack:tensor:0gru_7/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_7/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_8/ShapeShapegru_7/transpose_1:y:0*
T0*
_output_shapes
:c
gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_8/strided_sliceStridedSlicegru_8/Shape:output:0"gru_8/strided_slice/stack:output:0$gru_8/strided_slice/stack_1:output:0$gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_8/zeros/packedPackgru_8/strided_slice:output:0gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_8/zerosFillgru_8/zeros/packed:output:0gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_8/transpose	Transposegru_7/transpose_1:y:0gru_8/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_8/Shape_1Shapegru_8/transpose:y:0*
T0*
_output_shapes
:e
gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_8/strided_slice_1StridedSlicegru_8/Shape_1:output:0$gru_8/strided_slice_1/stack:output:0&gru_8/strided_slice_1/stack_1:output:0&gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_8/TensorArrayV2TensorListReserve*gru_8/TensorArrayV2/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_8/transpose:y:0Dgru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_8/strided_slice_2StridedSlicegru_8/transpose:y:0$gru_8/strided_slice_2/stack:output:0&gru_8/strided_slice_2/stack_1:output:0&gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
gru_8/gru_cell_8/ReadVariableOpReadVariableOp(gru_8_gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_8/gru_cell_8/unstackUnpack'gru_8/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_8/gru_cell_8/MatMul/ReadVariableOpReadVariableOp/gru_8_gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¤
gru_8/gru_cell_8/MatMulMatMulgru_8/strided_slice_2:output:0.gru_8/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_8/gru_cell_8/BiasAddBiasAdd!gru_8/gru_cell_8/MatMul:product:0!gru_8/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_8/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_8/gru_cell_8/splitSplit)gru_8/gru_cell_8/split/split_dim:output:0!gru_8/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_8/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp1gru_8_gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_8/gru_cell_8/MatMul_1MatMulgru_8/zeros:output:00gru_8/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_8/gru_cell_8/BiasAdd_1BiasAdd#gru_8/gru_cell_8/MatMul_1:product:0!gru_8/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_8/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_8/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_8/gru_cell_8/split_1SplitV#gru_8/gru_cell_8/BiasAdd_1:output:0gru_8/gru_cell_8/Const:output:0+gru_8/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_8/gru_cell_8/addAddV2gru_8/gru_cell_8/split:output:0!gru_8/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_8/gru_cell_8/SigmoidSigmoidgru_8/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/add_1AddV2gru_8/gru_cell_8/split:output:1!gru_8/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_8/gru_cell_8/Sigmoid_1Sigmoidgru_8/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mulMulgru_8/gru_cell_8/Sigmoid_1:y:0!gru_8/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/add_2AddV2gru_8/gru_cell_8/split:output:2gru_8/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_8/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/gru_cell_8/mul_1Mulgru_8/gru_cell_8/beta:output:0gru_8/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_8/gru_cell_8/Sigmoid_2Sigmoidgru_8/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mul_2Mulgru_8/gru_cell_8/add_2:z:0gru_8/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_8/gru_cell_8/IdentityIdentitygru_8/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
gru_8/gru_cell_8/IdentityN	IdentityNgru_8/gru_cell_8/mul_2:z:0gru_8/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48527*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mul_3Mulgru_8/gru_cell_8/Sigmoid:y:0gru_8/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_8/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/gru_cell_8/subSubgru_8/gru_cell_8/sub/x:output:0gru_8/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/mul_4Mulgru_8/gru_cell_8/sub:z:0#gru_8/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/gru_cell_8/add_3AddV2gru_8/gru_cell_8/mul_3:z:0gru_8/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_8/TensorArrayV2_1TensorListReserve,gru_8/TensorArrayV2_1/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_8/whileWhile!gru_8/while/loop_counter:output:0'gru_8/while/maximum_iterations:output:0gru_8/time:output:0gru_8/TensorArrayV2_1:handle:0gru_8/zeros:output:0gru_8/strided_slice_1:output:0=gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_8_gru_cell_8_readvariableop_resource/gru_8_gru_cell_8_matmul_readvariableop_resource1gru_8_gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *"
bodyR
gru_8_while_body_48543*"
condR
gru_8_while_cond_48542*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_8/TensorArrayV2Stack/TensorListStackTensorListStackgru_8/while:output:3?gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_8/strided_slice_3StridedSlice1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0$gru_8/strided_slice_3/stack:output:0&gru_8/strided_slice_3/stack_1:output:0&gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_8/transpose_1	Transpose1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0gru_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_2/MatMulMatMulgru_8/strided_slice_3:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^gru_6/gru_cell_6/MatMul/ReadVariableOp)^gru_6/gru_cell_6/MatMul_1/ReadVariableOp ^gru_6/gru_cell_6/ReadVariableOp^gru_6/while'^gru_7/gru_cell_7/MatMul/ReadVariableOp)^gru_7/gru_cell_7/MatMul_1/ReadVariableOp ^gru_7/gru_cell_7/ReadVariableOp^gru_7/while'^gru_8/gru_cell_8/MatMul/ReadVariableOp)^gru_8/gru_cell_8/MatMul_1/ReadVariableOp ^gru_8/gru_cell_8/ReadVariableOp^gru_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
&gru_6/gru_cell_6/MatMul/ReadVariableOp&gru_6/gru_cell_6/MatMul/ReadVariableOp2T
(gru_6/gru_cell_6/MatMul_1/ReadVariableOp(gru_6/gru_cell_6/MatMul_1/ReadVariableOp2B
gru_6/gru_cell_6/ReadVariableOpgru_6/gru_cell_6/ReadVariableOp2
gru_6/whilegru_6/while2P
&gru_7/gru_cell_7/MatMul/ReadVariableOp&gru_7/gru_cell_7/MatMul/ReadVariableOp2T
(gru_7/gru_cell_7/MatMul_1/ReadVariableOp(gru_7/gru_cell_7/MatMul_1/ReadVariableOp2B
gru_7/gru_cell_7/ReadVariableOpgru_7/gru_cell_7/ReadVariableOp2
gru_7/whilegru_7/while2P
&gru_8/gru_cell_8/MatMul/ReadVariableOp&gru_8/gru_cell_8/MatMul/ReadVariableOp2T
(gru_8/gru_cell_8/MatMul_1/ReadVariableOp(gru_8/gru_cell_8/MatMul_1/ReadVariableOp2B
gru_8/gru_cell_8/ReadVariableOpgru_8/gru_cell_8/ReadVariableOp2
gru_8/whilegru_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ý

"__inference_internal_grad_fn_52744
result_grads_0
result_grads_1
mul_while_gru_cell_8_beta
mul_while_gru_cell_8_add_2
identity
mulMulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_8_betamul_while_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
°

Ù
*__inference_gru_cell_7_layer_call_fn_51476

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46290o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
°

Ù
*__inference_gru_cell_8_layer_call_fn_51596

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0
å
§
while_body_46694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_gru_cell_8_46716_0:	¬+
while_gru_cell_8_46718_0:	d¬+
while_gru_cell_8_46720_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_gru_cell_8_46716:	¬)
while_gru_cell_8_46718:	d¬)
while_gru_cell_8_46720:	d¬¢(while/gru_cell_8/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0û
(while/gru_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_8_46716_0while_gru_cell_8_46718_0while_gru_cell_8_46720_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46642Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_8_46716while_gru_cell_8_46716_0"2
while_gru_cell_8_46718while_gru_cell_8_46718_0"2
while_gru_cell_8_46720while_gru_cell_8_46720_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/gru_cell_8/StatefulPartitionedCall(while/gru_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ý

"__inference_internal_grad_fn_52564
result_grads_0
result_grads_1
mul_while_gru_cell_7_beta
mul_while_gru_cell_7_add_2
identity
mulMulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_7_betamul_while_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ËQ

@__inference_gru_8_layer_call_and_return_conditional_losses_51142

inputs5
"gru_cell_8_readvariableop_resource:	¬<
)gru_cell_8_matmul_readvariableop_resource:	d¬>
+gru_cell_8_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_8/MatMul/ReadVariableOp¢"gru_cell_8/MatMul_1/ReadVariableOp¢gru_cell_8/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_8/ReadVariableOpReadVariableOp"gru_cell_8_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_8/unstackUnpack!gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_8/MatMul/ReadVariableOpReadVariableOp)gru_cell_8_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMulMatMulstrided_slice_2:output:0(gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAddBiasAddgru_cell_8/MatMul:product:0gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_8/splitSplit#gru_cell_8/split/split_dim:output:0gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_8/MatMul_1MatMulzeros:output:0*gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_8/BiasAdd_1BiasAddgru_cell_8/MatMul_1:product:0gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_8/split_1SplitVgru_cell_8/BiasAdd_1:output:0gru_cell_8/Const:output:0%gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_8/addAddV2gru_cell_8/split:output:0gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_8/SigmoidSigmoidgru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_8/add_1AddV2gru_cell_8/split:output:1gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_1Sigmoidgru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_8/mulMulgru_cell_8/Sigmoid_1:y:0gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_8/add_2AddV2gru_cell_8/split:output:2gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_8/mul_1Mulgru_cell_8/beta:output:0gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/Sigmoid_2Sigmoidgru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_8/mul_2Mulgru_cell_8/add_2:z:0gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_8/IdentityIdentitygru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_8/IdentityN	IdentityNgru_cell_8/mul_2:z:0gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-51030*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_8/mul_3Mulgru_cell_8/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_8/subSubgru_cell_8/sub/x:output:0gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_8/mul_4Mulgru_cell_8/sub:z:0gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_8/add_3AddV2gru_cell_8/mul_3:z:0gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_8_readvariableop_resource)gru_cell_8_matmul_readvariableop_resource+gru_cell_8_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_51046*
condR
while_cond_51045*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_8/MatMul/ReadVariableOp#^gru_cell_8/MatMul_1/ReadVariableOp^gru_cell_8/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_8/MatMul/ReadVariableOp gru_cell_8/MatMul/ReadVariableOp2H
"gru_cell_8/MatMul_1/ReadVariableOp"gru_cell_8/MatMul_1/ReadVariableOp26
gru_cell_8/ReadVariableOpgru_cell_8/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ã¢

!__inference__traced_restore_53149
file_prefix1
assignvariableop_dense_2_kernel:d-
assignvariableop_1_dense_2_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
*assignvariableop_7_gru_6_gru_cell_6_kernel:	¬G
4assignvariableop_8_gru_6_gru_cell_6_recurrent_kernel:	d¬;
(assignvariableop_9_gru_6_gru_cell_6_bias:	¬>
+assignvariableop_10_gru_7_gru_cell_7_kernel:	d¬H
5assignvariableop_11_gru_7_gru_cell_7_recurrent_kernel:	d¬<
)assignvariableop_12_gru_7_gru_cell_7_bias:	¬>
+assignvariableop_13_gru_8_gru_cell_8_kernel:	d¬H
5assignvariableop_14_gru_8_gru_cell_8_recurrent_kernel:	d¬<
)assignvariableop_15_gru_8_gru_cell_8_bias:	¬#
assignvariableop_16_total: #
assignvariableop_17_count: ;
)assignvariableop_18_adam_dense_2_kernel_m:d5
'assignvariableop_19_adam_dense_2_bias_m:E
2assignvariableop_20_adam_gru_6_gru_cell_6_kernel_m:	¬O
<assignvariableop_21_adam_gru_6_gru_cell_6_recurrent_kernel_m:	d¬C
0assignvariableop_22_adam_gru_6_gru_cell_6_bias_m:	¬E
2assignvariableop_23_adam_gru_7_gru_cell_7_kernel_m:	d¬O
<assignvariableop_24_adam_gru_7_gru_cell_7_recurrent_kernel_m:	d¬C
0assignvariableop_25_adam_gru_7_gru_cell_7_bias_m:	¬E
2assignvariableop_26_adam_gru_8_gru_cell_8_kernel_m:	d¬O
<assignvariableop_27_adam_gru_8_gru_cell_8_recurrent_kernel_m:	d¬C
0assignvariableop_28_adam_gru_8_gru_cell_8_bias_m:	¬;
)assignvariableop_29_adam_dense_2_kernel_v:d5
'assignvariableop_30_adam_dense_2_bias_v:E
2assignvariableop_31_adam_gru_6_gru_cell_6_kernel_v:	¬O
<assignvariableop_32_adam_gru_6_gru_cell_6_recurrent_kernel_v:	d¬C
0assignvariableop_33_adam_gru_6_gru_cell_6_bias_v:	¬E
2assignvariableop_34_adam_gru_7_gru_cell_7_kernel_v:	d¬O
<assignvariableop_35_adam_gru_7_gru_cell_7_recurrent_kernel_v:	d¬C
0assignvariableop_36_adam_gru_7_gru_cell_7_bias_v:	¬E
2assignvariableop_37_adam_gru_8_gru_cell_8_kernel_v:	d¬O
<assignvariableop_38_adam_gru_8_gru_cell_8_recurrent_kernel_v:	d¬C
0assignvariableop_39_adam_gru_8_gru_cell_8_bias_v:	¬
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9È
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*î
valueäBá)B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B î
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp*assignvariableop_7_gru_6_gru_cell_6_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_8AssignVariableOp4assignvariableop_8_gru_6_gru_cell_6_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp(assignvariableop_9_gru_6_gru_cell_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp+assignvariableop_10_gru_7_gru_cell_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_11AssignVariableOp5assignvariableop_11_gru_7_gru_cell_7_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp)assignvariableop_12_gru_7_gru_cell_7_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp+assignvariableop_13_gru_8_gru_cell_8_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_14AssignVariableOp5assignvariableop_14_gru_8_gru_cell_8_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp)assignvariableop_15_gru_8_gru_cell_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_2_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_2_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_gru_6_gru_cell_6_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_gru_6_gru_cell_6_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_gru_6_gru_cell_6_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_gru_7_gru_cell_7_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_gru_7_gru_cell_7_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_gru_7_gru_cell_7_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_gru_8_gru_cell_8_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_27AssignVariableOp<assignvariableop_27_adam_gru_8_gru_cell_8_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_gru_8_gru_cell_8_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_gru_6_gru_cell_6_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_32AssignVariableOp<assignvariableop_32_adam_gru_6_gru_cell_6_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_gru_6_gru_cell_6_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_gru_7_gru_cell_7_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adam_gru_7_gru_cell_7_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_gru_7_gru_cell_7_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_gru_8_gru_cell_8_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_38AssignVariableOp<assignvariableop_38_adam_gru_8_gru_cell_8_recurrent_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_gru_8_gru_cell_8_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¿
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: ¬
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ý

"__inference_internal_grad_fn_52024
result_grads_0
result_grads_1
mul_while_gru_cell_6_beta
mul_while_gru_cell_6_add_2
identity
mulMulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ÆQ

@__inference_gru_7_layer_call_and_return_conditional_losses_50430

inputs5
"gru_cell_7_readvariableop_resource:	¬<
)gru_cell_7_matmul_readvariableop_resource:	d¬>
+gru_cell_7_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_7/MatMul/ReadVariableOp¢"gru_cell_7/MatMul_1/ReadVariableOp¢gru_cell_7/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_7/ReadVariableOpReadVariableOp"gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_7/unstackUnpack!gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_7/MatMul/ReadVariableOpReadVariableOp)gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMulMatMulstrided_slice_2:output:0(gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAddBiasAddgru_cell_7/MatMul:product:0gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_7/splitSplit#gru_cell_7/split/split_dim:output:0gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMul_1MatMulzeros:output:0*gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAdd_1BiasAddgru_cell_7/MatMul_1:product:0gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_7/split_1SplitVgru_cell_7/BiasAdd_1:output:0gru_cell_7/Const:output:0%gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_7/addAddV2gru_cell_7/split:output:0gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_7/SigmoidSigmoidgru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_7/add_1AddV2gru_cell_7/split:output:1gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_1Sigmoidgru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_7/mulMulgru_cell_7/Sigmoid_1:y:0gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_7/add_2AddV2gru_cell_7/split:output:2gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_7/mul_1Mulgru_cell_7/beta:output:0gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_2Sigmoidgru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_7/mul_2Mulgru_cell_7/add_2:z:0gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/IdentityIdentitygru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_7/IdentityN	IdentityNgru_cell_7/mul_2:z:0gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50318*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_7/mul_3Mulgru_cell_7/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_7/subSubgru_cell_7/sub/x:output:0gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_7/mul_4Mulgru_cell_7/sub:z:0gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_7/add_3AddV2gru_cell_7/mul_3:z:0gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_7_readvariableop_resource)gru_cell_7_matmul_readvariableop_resource+gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_50334*
condR
while_cond_50333*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd²
NoOpNoOp!^gru_cell_7/MatMul/ReadVariableOp#^gru_cell_7/MatMul_1/ReadVariableOp^gru_cell_7/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2D
 gru_cell_7/MatMul/ReadVariableOp gru_cell_7/MatMul/ReadVariableOp2H
"gru_cell_7/MatMul_1/ReadVariableOp"gru_cell_7/MatMul_1/ReadVariableOp26
gru_cell_7/ReadVariableOpgru_cell_7/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
J
²	
gru_8_while_body_48543(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2'
#gru_8_while_gru_8_strided_slice_1_0c
_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0C
0gru_8_while_gru_cell_8_readvariableop_resource_0:	¬J
7gru_8_while_gru_cell_8_matmul_readvariableop_resource_0:	d¬L
9gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0:	d¬
gru_8_while_identity
gru_8_while_identity_1
gru_8_while_identity_2
gru_8_while_identity_3
gru_8_while_identity_4%
!gru_8_while_gru_8_strided_slice_1a
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensorA
.gru_8_while_gru_cell_8_readvariableop_resource:	¬H
5gru_8_while_gru_cell_8_matmul_readvariableop_resource:	d¬J
7gru_8_while_gru_cell_8_matmul_1_readvariableop_resource:	d¬¢,gru_8/while/gru_cell_8/MatMul/ReadVariableOp¢.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp¢%gru_8/while/gru_cell_8/ReadVariableOp
=gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0gru_8_while_placeholderFgru_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
%gru_8/while/gru_cell_8/ReadVariableOpReadVariableOp0gru_8_while_gru_cell_8_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_8/while/gru_cell_8/unstackUnpack-gru_8/while/gru_cell_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_8/while/gru_cell_8/MatMul/ReadVariableOpReadVariableOp7gru_8_while_gru_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0È
gru_8/while/gru_cell_8/MatMulMatMul6gru_8/while/TensorArrayV2Read/TensorListGetItem:item:04gru_8/while/gru_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_8/while/gru_cell_8/BiasAddBiasAdd'gru_8/while/gru_cell_8/MatMul:product:0'gru_8/while/gru_cell_8/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_8/while/gru_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_8/while/gru_cell_8/splitSplit/gru_8/while/gru_cell_8/split/split_dim:output:0'gru_8/while/gru_cell_8/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOpReadVariableOp9gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_8/while/gru_cell_8/MatMul_1MatMulgru_8_while_placeholder_26gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_8/while/gru_cell_8/BiasAdd_1BiasAdd)gru_8/while/gru_cell_8/MatMul_1:product:0'gru_8/while/gru_cell_8/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_8/while/gru_cell_8/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_8/while/gru_cell_8/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_8/while/gru_cell_8/split_1SplitV)gru_8/while/gru_cell_8/BiasAdd_1:output:0%gru_8/while/gru_cell_8/Const:output:01gru_8/while/gru_cell_8/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_8/while/gru_cell_8/addAddV2%gru_8/while/gru_cell_8/split:output:0'gru_8/while/gru_cell_8/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_8/while/gru_cell_8/SigmoidSigmoidgru_8/while/gru_cell_8/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_8/while/gru_cell_8/add_1AddV2%gru_8/while/gru_cell_8/split:output:1'gru_8/while/gru_cell_8/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_8/while/gru_cell_8/Sigmoid_1Sigmoid gru_8/while/gru_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_8/while/gru_cell_8/mulMul$gru_8/while/gru_cell_8/Sigmoid_1:y:0'gru_8/while/gru_cell_8/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/add_2AddV2%gru_8/while/gru_cell_8/split:output:2gru_8/while/gru_cell_8/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_8/while/gru_cell_8/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/while/gru_cell_8/mul_1Mul$gru_8/while/gru_cell_8/beta:output:0 gru_8/while/gru_cell_8/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_8/while/gru_cell_8/Sigmoid_2Sigmoid gru_8/while/gru_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/mul_2Mul gru_8/while/gru_cell_8/add_2:z:0$gru_8/while/gru_cell_8/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/IdentityIdentity gru_8/while/gru_cell_8/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdç
 gru_8/while/gru_cell_8/IdentityN	IdentityN gru_8/while/gru_cell_8/mul_2:z:0 gru_8/while/gru_cell_8/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-48593*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/mul_3Mul"gru_8/while/gru_cell_8/Sigmoid:y:0gru_8_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_8/while/gru_cell_8/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/while/gru_cell_8/subSub%gru_8/while/gru_cell_8/sub/x:output:0"gru_8/while/gru_cell_8/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_8/while/gru_cell_8/mul_4Mulgru_8/while/gru_cell_8/sub:z:0)gru_8/while/gru_cell_8/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_8/while/gru_cell_8/add_3AddV2 gru_8/while/gru_cell_8/mul_3:z:0 gru_8/while/gru_cell_8/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_8_while_placeholder_1gru_8_while_placeholder gru_8/while/gru_cell_8/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_8/while/addAddV2gru_8_while_placeholdergru_8/while/add/y:output:0*
T0*
_output_shapes
: U
gru_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_8/while/add_1AddV2$gru_8_while_gru_8_while_loop_countergru_8/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_8/while/IdentityIdentitygru_8/while/add_1:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: 
gru_8/while/Identity_1Identity*gru_8_while_gru_8_while_maximum_iterations^gru_8/while/NoOp*
T0*
_output_shapes
: k
gru_8/while/Identity_2Identitygru_8/while/add:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: «
gru_8/while/Identity_3Identity@gru_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_8/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_8/while/Identity_4Identity gru_8/while/gru_cell_8/add_3:z:0^gru_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_8/while/NoOpNoOp-^gru_8/while/gru_cell_8/MatMul/ReadVariableOp/^gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp&^gru_8/while/gru_cell_8/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_8_while_gru_8_strided_slice_1#gru_8_while_gru_8_strided_slice_1_0"t
7gru_8_while_gru_cell_8_matmul_1_readvariableop_resource9gru_8_while_gru_cell_8_matmul_1_readvariableop_resource_0"p
5gru_8_while_gru_cell_8_matmul_readvariableop_resource7gru_8_while_gru_cell_8_matmul_readvariableop_resource_0"b
.gru_8_while_gru_cell_8_readvariableop_resource0gru_8_while_gru_cell_8_readvariableop_resource_0"5
gru_8_while_identitygru_8/while/Identity:output:0"9
gru_8_while_identity_1gru_8/while/Identity_1:output:0"9
gru_8_while_identity_2gru_8/while/Identity_2:output:0"9
gru_8_while_identity_3gru_8/while/Identity_3:output:0"9
gru_8_while_identity_4gru_8/while/Identity_4:output:0"À
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_8/while/gru_cell_8/MatMul/ReadVariableOp,gru_8/while/gru_cell_8/MatMul/ReadVariableOp2`
.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp.gru_8/while/gru_cell_8/MatMul_1/ReadVariableOp2N
%gru_8/while/gru_cell_8/ReadVariableOp%gru_8/while/gru_cell_8/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
¾

'__inference_dense_2_layer_call_fn_51318

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_47306o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
!
Ù
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45788

inputs

states*
readvariableop_resource:	¬1
matmul_readvariableop_resource:	¬3
 matmul_1_readvariableop_resource:	d¬

identity_1

identity_2¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	¬*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
mul_1Mulbeta:output:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
	Sigmoid_2Sigmoid	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
mul_2Mul	add_2:z:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-45774*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdS
mul_3MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
mul_4Mulsub:z:0IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
add_3AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
Õ
¥
while_cond_47809
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_47809___redundant_placeholder03
/while_while_cond_47809___redundant_placeholder13
/while_while_cond_47809___redundant_placeholder23
/while_while_cond_47809___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ø

"__inference_internal_grad_fn_52438
result_grads_0
result_grads_1
mul_gru_cell_6_beta
mul_gru_cell_6_add_2
identityx
mulMulmul_gru_cell_6_betamul_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_6_betamul_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
R

@__inference_gru_7_layer_call_and_return_conditional_losses_50263
inputs_05
"gru_cell_7_readvariableop_resource:	¬<
)gru_cell_7_matmul_readvariableop_resource:	d¬>
+gru_cell_7_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_7/MatMul/ReadVariableOp¢"gru_cell_7/MatMul_1/ReadVariableOp¢gru_cell_7/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask}
gru_cell_7/ReadVariableOpReadVariableOp"gru_cell_7_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_7/unstackUnpack!gru_cell_7/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_7/MatMul/ReadVariableOpReadVariableOp)gru_cell_7_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMulMatMulstrided_slice_2:output:0(gru_cell_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAddBiasAddgru_cell_7/MatMul:product:0gru_cell_7/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_7/splitSplit#gru_cell_7/split/split_dim:output:0gru_cell_7/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_7/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_7_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_7/MatMul_1MatMulzeros:output:0*gru_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_7/BiasAdd_1BiasAddgru_cell_7/MatMul_1:product:0gru_cell_7/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_7/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_7/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_7/split_1SplitVgru_cell_7/BiasAdd_1:output:0gru_cell_7/Const:output:0%gru_cell_7/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_7/addAddV2gru_cell_7/split:output:0gru_cell_7/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_7/SigmoidSigmoidgru_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_7/add_1AddV2gru_cell_7/split:output:1gru_cell_7/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_1Sigmoidgru_cell_7/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_7/mulMulgru_cell_7/Sigmoid_1:y:0gru_cell_7/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_7/add_2AddV2gru_cell_7/split:output:2gru_cell_7/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_7/mul_1Mulgru_cell_7/beta:output:0gru_cell_7/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/Sigmoid_2Sigmoidgru_cell_7/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_7/mul_2Mulgru_cell_7/add_2:z:0gru_cell_7/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_7/IdentityIdentitygru_cell_7/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
gru_cell_7/IdentityN	IdentityNgru_cell_7/mul_2:z:0gru_cell_7/add_2:z:0*
T
2*+
_gradient_op_typeCustomGradient-50151*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_7/mul_3Mulgru_cell_7/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_7/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_7/subSubgru_cell_7/sub/x:output:0gru_cell_7/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_7/mul_4Mulgru_cell_7/sub:z:0gru_cell_7/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_7/add_3AddV2gru_cell_7/mul_3:z:0gru_cell_7/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_7_readvariableop_resource)gru_cell_7_matmul_readvariableop_resource+gru_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_50167*
condR
while_cond_50166*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd²
NoOpNoOp!^gru_cell_7/MatMul/ReadVariableOp#^gru_cell_7/MatMul_1/ReadVariableOp^gru_cell_7/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_7/MatMul/ReadVariableOp gru_cell_7/MatMul/ReadVariableOp2H
"gru_cell_7/MatMul_1/ReadVariableOp"gru_cell_7/MatMul_1/ReadVariableOp26
gru_cell_7/ReadVariableOpgru_cell_7/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
Ø

"__inference_internal_grad_fn_51970
result_grads_0
result_grads_1
mul_gru_cell_8_beta
mul_gru_cell_8_add_2
identityx
mulMulmul_gru_cell_8_betamul_gru_cell_8_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_8_betamul_gru_cell_8_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
ý

"__inference_internal_grad_fn_52420
result_grads_0
result_grads_1
mul_while_gru_cell_6_beta
mul_while_gru_cell_6_add_2
identity
mulMulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_6_betamul_while_gru_cell_6_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ø

"__inference_internal_grad_fn_52546
result_grads_0
result_grads_1
mul_gru_cell_7_beta
mul_gru_cell_7_add_2
identityx
mulMulmul_gru_cell_7_betamul_gru_cell_7_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_7_betamul_gru_cell_7_add_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: :ÿÿÿÿÿÿÿÿÿd:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:
"__inference_internal_grad_fn_51790CustomGradient-45267:
"__inference_internal_grad_fn_51808CustomGradient-45430:
"__inference_internal_grad_fn_51826CustomGradient-45593:
"__inference_internal_grad_fn_51844CustomGradient-45333:
"__inference_internal_grad_fn_51862CustomGradient-45496:
"__inference_internal_grad_fn_51880CustomGradient-45659:
"__inference_internal_grad_fn_51898CustomGradient-46828:
"__inference_internal_grad_fn_51916CustomGradient-46894:
"__inference_internal_grad_fn_51934CustomGradient-47002:
"__inference_internal_grad_fn_51952CustomGradient-47068:
"__inference_internal_grad_fn_51970CustomGradient-47176:
"__inference_internal_grad_fn_51988CustomGradient-47242:
"__inference_internal_grad_fn_52006CustomGradient-47794:
"__inference_internal_grad_fn_52024CustomGradient-47860:
"__inference_internal_grad_fn_52042CustomGradient-47605:
"__inference_internal_grad_fn_52060CustomGradient-47671:
"__inference_internal_grad_fn_52078CustomGradient-47416:
"__inference_internal_grad_fn_52096CustomGradient-47482:
"__inference_internal_grad_fn_52114CustomGradient-48201:
"__inference_internal_grad_fn_52132CustomGradient-48364:
"__inference_internal_grad_fn_52150CustomGradient-48527:
"__inference_internal_grad_fn_52168CustomGradient-48267:
"__inference_internal_grad_fn_52186CustomGradient-48430:
"__inference_internal_grad_fn_52204CustomGradient-48593:
"__inference_internal_grad_fn_52222CustomGradient-48700:
"__inference_internal_grad_fn_52240CustomGradient-48863:
"__inference_internal_grad_fn_52258CustomGradient-49026:
"__inference_internal_grad_fn_52276CustomGradient-48766:
"__inference_internal_grad_fn_52294CustomGradient-48929:
"__inference_internal_grad_fn_52312CustomGradient-49092:
"__inference_internal_grad_fn_52330CustomGradient-45774:
"__inference_internal_grad_fn_52348CustomGradient-45924:
"__inference_internal_grad_fn_52366CustomGradient-49272:
"__inference_internal_grad_fn_52384CustomGradient-49338:
"__inference_internal_grad_fn_52402CustomGradient-49439:
"__inference_internal_grad_fn_52420CustomGradient-49505:
"__inference_internal_grad_fn_52438CustomGradient-49606:
"__inference_internal_grad_fn_52456CustomGradient-49672:
"__inference_internal_grad_fn_52474CustomGradient-49773:
"__inference_internal_grad_fn_52492CustomGradient-49839:
"__inference_internal_grad_fn_52510CustomGradient-46126:
"__inference_internal_grad_fn_52528CustomGradient-46276:
"__inference_internal_grad_fn_52546CustomGradient-49984:
"__inference_internal_grad_fn_52564CustomGradient-50050:
"__inference_internal_grad_fn_52582CustomGradient-50151:
"__inference_internal_grad_fn_52600CustomGradient-50217:
"__inference_internal_grad_fn_52618CustomGradient-50318:
"__inference_internal_grad_fn_52636CustomGradient-50384:
"__inference_internal_grad_fn_52654CustomGradient-50485:
"__inference_internal_grad_fn_52672CustomGradient-50551:
"__inference_internal_grad_fn_52690CustomGradient-46478:
"__inference_internal_grad_fn_52708CustomGradient-46628:
"__inference_internal_grad_fn_52726CustomGradient-50696:
"__inference_internal_grad_fn_52744CustomGradient-50762:
"__inference_internal_grad_fn_52762CustomGradient-50863:
"__inference_internal_grad_fn_52780CustomGradient-50929:
"__inference_internal_grad_fn_52798CustomGradient-51030:
"__inference_internal_grad_fn_52816CustomGradient-51096:
"__inference_internal_grad_fn_52834CustomGradient-51197:
"__inference_internal_grad_fn_52852CustomGradient-51263:
"__inference_internal_grad_fn_52870CustomGradient-51388:
"__inference_internal_grad_fn_52888CustomGradient-51434:
"__inference_internal_grad_fn_52906CustomGradient-51508:
"__inference_internal_grad_fn_52924CustomGradient-51554:
"__inference_internal_grad_fn_52942CustomGradient-51628:
"__inference_internal_grad_fn_52960CustomGradient-51674"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
G
gru_6_input8
serving_default_gru_6_input:0ÿÿÿÿÿÿÿÿÿd;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ò

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Ú
 cell
!
state_spec
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&_random_generator
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
»

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
¯
1iter

2beta_1

3beta_2
	4decay
5learning_rate)m*m6m7m8m9m:m;m<m=m>m)v*v6v7v8v9v:v;v<v=v>v"
	optimizer
n
60
71
82
93
:4
;5
<6
=7
>8
)9
*10"
trackable_list_wrapper
n
60
71
82
93
:4
;5
<6
=7
>8
)9
*10"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_2_layer_call_fn_47338
,__inference_sequential_2_layer_call_fn_48119
,__inference_sequential_2_layer_call_fn_48146
,__inference_sequential_2_layer_call_fn_48026À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
G__inference_sequential_2_layer_call_and_return_conditional_losses_48056
G__inference_sequential_2_layer_call_and_return_conditional_losses_48086À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÏBÌ
 __inference__wrapped_model_45711gru_6_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Dserving_default"
signature_map
è

6kernel
7recurrent_kernel
8bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I_random_generator
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Lstates
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
÷2ô
%__inference_gru_6_layer_call_fn_49184
%__inference_gru_6_layer_call_fn_49195
%__inference_gru_6_layer_call_fn_49206
%__inference_gru_6_layer_call_fn_49217Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
@__inference_gru_6_layer_call_and_return_conditional_losses_49384
@__inference_gru_6_layer_call_and_return_conditional_losses_49551
@__inference_gru_6_layer_call_and_return_conditional_losses_49718
@__inference_gru_6_layer_call_and_return_conditional_losses_49885Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è

9kernel
:recurrent_kernel
;bias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
÷2ô
%__inference_gru_7_layer_call_fn_49896
%__inference_gru_7_layer_call_fn_49907
%__inference_gru_7_layer_call_fn_49918
%__inference_gru_7_layer_call_fn_49929Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
@__inference_gru_7_layer_call_and_return_conditional_losses_50096
@__inference_gru_7_layer_call_and_return_conditional_losses_50263
@__inference_gru_7_layer_call_and_return_conditional_losses_50430
@__inference_gru_7_layer_call_and_return_conditional_losses_50597Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è

<kernel
=recurrent_kernel
>bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c_random_generator
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

fstates
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
"	variables
#trainable_variables
$regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
÷2ô
%__inference_gru_8_layer_call_fn_50608
%__inference_gru_8_layer_call_fn_50619
%__inference_gru_8_layer_call_fn_50630
%__inference_gru_8_layer_call_fn_50641Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
@__inference_gru_8_layer_call_and_return_conditional_losses_50808
@__inference_gru_8_layer_call_and_return_conditional_losses_50975
@__inference_gru_8_layer_call_and_return_conditional_losses_51142
@__inference_gru_8_layer_call_and_return_conditional_losses_51309Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :d2dense_2/kernel
:2dense_2/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_2_layer_call_fn_51318¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_2_layer_call_and_return_conditional_losses_51328¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	¬2gru_6/gru_cell_6/kernel
4:2	d¬2!gru_6/gru_cell_6/recurrent_kernel
(:&	¬2gru_6/gru_cell_6/bias
*:(	d¬2gru_7/gru_cell_7/kernel
4:2	d¬2!gru_7/gru_cell_7/recurrent_kernel
(:&	¬2gru_7/gru_cell_7/bias
*:(	d¬2gru_8/gru_cell_8/kernel
4:2	d¬2!gru_8/gru_cell_8/recurrent_kernel
(:&	¬2gru_8/gru_cell_8/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÎBË
#__inference_signature_wrapper_49173gru_6_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_gru_cell_6_layer_call_fn_51342
*__inference_gru_cell_6_layer_call_fn_51356¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51402
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51448¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_gru_cell_7_layer_call_fn_51462
*__inference_gru_cell_7_layer_call_fn_51476¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51522
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51568¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
®
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_gru_cell_8_layer_call_fn_51582
*__inference_gru_cell_8_layer_call_fn_51596¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51642
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51688¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
%:#d2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
/:-	¬2Adam/gru_6/gru_cell_6/kernel/m
9:7	d¬2(Adam/gru_6/gru_cell_6/recurrent_kernel/m
-:+	¬2Adam/gru_6/gru_cell_6/bias/m
/:-	d¬2Adam/gru_7/gru_cell_7/kernel/m
9:7	d¬2(Adam/gru_7/gru_cell_7/recurrent_kernel/m
-:+	¬2Adam/gru_7/gru_cell_7/bias/m
/:-	d¬2Adam/gru_8/gru_cell_8/kernel/m
9:7	d¬2(Adam/gru_8/gru_cell_8/recurrent_kernel/m
-:+	¬2Adam/gru_8/gru_cell_8/bias/m
%:#d2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
/:-	¬2Adam/gru_6/gru_cell_6/kernel/v
9:7	d¬2(Adam/gru_6/gru_cell_6/recurrent_kernel/v
-:+	¬2Adam/gru_6/gru_cell_6/bias/v
/:-	d¬2Adam/gru_7/gru_cell_7/kernel/v
9:7	d¬2(Adam/gru_7/gru_cell_7/recurrent_kernel/v
-:+	¬2Adam/gru_7/gru_cell_7/bias/v
/:-	d¬2Adam/gru_8/gru_cell_8/kernel/v
9:7	d¬2(Adam/gru_8/gru_cell_8/recurrent_kernel/v
-:+	¬2Adam/gru_8/gru_cell_8/bias/v
JbH
$sequential_2/gru_6/gru_cell_6/beta:0 __inference__wrapped_model_45711
KbI
%sequential_2/gru_6/gru_cell_6/add_2:0 __inference__wrapped_model_45711
JbH
$sequential_2/gru_7/gru_cell_7/beta:0 __inference__wrapped_model_45711
KbI
%sequential_2/gru_7/gru_cell_7/add_2:0 __inference__wrapped_model_45711
JbH
$sequential_2/gru_8/gru_cell_8/beta:0 __inference__wrapped_model_45711
KbI
%sequential_2/gru_8/gru_cell_8/add_2:0 __inference__wrapped_model_45711
SbQ
*sequential_2/gru_6/while/gru_cell_6/beta:0#sequential_2_gru_6_while_body_45283
TbR
+sequential_2/gru_6/while/gru_cell_6/add_2:0#sequential_2_gru_6_while_body_45283
SbQ
*sequential_2/gru_7/while/gru_cell_7/beta:0#sequential_2_gru_7_while_body_45446
TbR
+sequential_2/gru_7/while/gru_cell_7/add_2:0#sequential_2_gru_7_while_body_45446
SbQ
*sequential_2/gru_8/while/gru_cell_8/beta:0#sequential_2_gru_8_while_body_45609
TbR
+sequential_2/gru_8/while/gru_cell_8/add_2:0#sequential_2_gru_8_while_body_45609
WbU
gru_cell_6/beta:0@__inference_gru_6_layer_call_and_return_conditional_losses_46940
XbV
gru_cell_6/add_2:0@__inference_gru_6_layer_call_and_return_conditional_losses_46940
-b+
while/gru_cell_6/beta:0while_body_46844
.b,
while/gru_cell_6/add_2:0while_body_46844
WbU
gru_cell_7/beta:0@__inference_gru_7_layer_call_and_return_conditional_losses_47114
XbV
gru_cell_7/add_2:0@__inference_gru_7_layer_call_and_return_conditional_losses_47114
-b+
while/gru_cell_7/beta:0while_body_47018
.b,
while/gru_cell_7/add_2:0while_body_47018
WbU
gru_cell_8/beta:0@__inference_gru_8_layer_call_and_return_conditional_losses_47288
XbV
gru_cell_8/add_2:0@__inference_gru_8_layer_call_and_return_conditional_losses_47288
-b+
while/gru_cell_8/beta:0while_body_47192
.b,
while/gru_cell_8/add_2:0while_body_47192
WbU
gru_cell_6/beta:0@__inference_gru_6_layer_call_and_return_conditional_losses_47906
XbV
gru_cell_6/add_2:0@__inference_gru_6_layer_call_and_return_conditional_losses_47906
-b+
while/gru_cell_6/beta:0while_body_47810
.b,
while/gru_cell_6/add_2:0while_body_47810
WbU
gru_cell_7/beta:0@__inference_gru_7_layer_call_and_return_conditional_losses_47717
XbV
gru_cell_7/add_2:0@__inference_gru_7_layer_call_and_return_conditional_losses_47717
-b+
while/gru_cell_7/beta:0while_body_47621
.b,
while/gru_cell_7/add_2:0while_body_47621
WbU
gru_cell_8/beta:0@__inference_gru_8_layer_call_and_return_conditional_losses_47528
XbV
gru_cell_8/add_2:0@__inference_gru_8_layer_call_and_return_conditional_losses_47528
-b+
while/gru_cell_8/beta:0while_body_47432
.b,
while/gru_cell_8/add_2:0while_body_47432
dbb
gru_6/gru_cell_6/beta:0G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
ebc
gru_6/gru_cell_6/add_2:0G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
dbb
gru_7/gru_cell_7/beta:0G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
ebc
gru_7/gru_cell_7/add_2:0G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
dbb
gru_8/gru_cell_8/beta:0G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
ebc
gru_8/gru_cell_8/add_2:0G__inference_sequential_2_layer_call_and_return_conditional_losses_48645
9b7
gru_6/while/gru_cell_6/beta:0gru_6_while_body_48217
:b8
gru_6/while/gru_cell_6/add_2:0gru_6_while_body_48217
9b7
gru_7/while/gru_cell_7/beta:0gru_7_while_body_48380
:b8
gru_7/while/gru_cell_7/add_2:0gru_7_while_body_48380
9b7
gru_8/while/gru_cell_8/beta:0gru_8_while_body_48543
:b8
gru_8/while/gru_cell_8/add_2:0gru_8_while_body_48543
dbb
gru_6/gru_cell_6/beta:0G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
ebc
gru_6/gru_cell_6/add_2:0G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
dbb
gru_7/gru_cell_7/beta:0G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
ebc
gru_7/gru_cell_7/add_2:0G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
dbb
gru_8/gru_cell_8/beta:0G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
ebc
gru_8/gru_cell_8/add_2:0G__inference_sequential_2_layer_call_and_return_conditional_losses_49144
9b7
gru_6/while/gru_cell_6/beta:0gru_6_while_body_48716
:b8
gru_6/while/gru_cell_6/add_2:0gru_6_while_body_48716
9b7
gru_7/while/gru_cell_7/beta:0gru_7_while_body_48879
:b8
gru_7/while/gru_cell_7/add_2:0gru_7_while_body_48879
9b7
gru_8/while/gru_cell_8/beta:0gru_8_while_body_49042
:b8
gru_8/while/gru_cell_8/add_2:0gru_8_while_body_49042
QbO
beta:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45788
RbP
add_2:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45788
QbO
beta:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45938
RbP
add_2:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_45938
WbU
gru_cell_6/beta:0@__inference_gru_6_layer_call_and_return_conditional_losses_49384
XbV
gru_cell_6/add_2:0@__inference_gru_6_layer_call_and_return_conditional_losses_49384
-b+
while/gru_cell_6/beta:0while_body_49288
.b,
while/gru_cell_6/add_2:0while_body_49288
WbU
gru_cell_6/beta:0@__inference_gru_6_layer_call_and_return_conditional_losses_49551
XbV
gru_cell_6/add_2:0@__inference_gru_6_layer_call_and_return_conditional_losses_49551
-b+
while/gru_cell_6/beta:0while_body_49455
.b,
while/gru_cell_6/add_2:0while_body_49455
WbU
gru_cell_6/beta:0@__inference_gru_6_layer_call_and_return_conditional_losses_49718
XbV
gru_cell_6/add_2:0@__inference_gru_6_layer_call_and_return_conditional_losses_49718
-b+
while/gru_cell_6/beta:0while_body_49622
.b,
while/gru_cell_6/add_2:0while_body_49622
WbU
gru_cell_6/beta:0@__inference_gru_6_layer_call_and_return_conditional_losses_49885
XbV
gru_cell_6/add_2:0@__inference_gru_6_layer_call_and_return_conditional_losses_49885
-b+
while/gru_cell_6/beta:0while_body_49789
.b,
while/gru_cell_6/add_2:0while_body_49789
QbO
beta:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46140
RbP
add_2:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46140
QbO
beta:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46290
RbP
add_2:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_46290
WbU
gru_cell_7/beta:0@__inference_gru_7_layer_call_and_return_conditional_losses_50096
XbV
gru_cell_7/add_2:0@__inference_gru_7_layer_call_and_return_conditional_losses_50096
-b+
while/gru_cell_7/beta:0while_body_50000
.b,
while/gru_cell_7/add_2:0while_body_50000
WbU
gru_cell_7/beta:0@__inference_gru_7_layer_call_and_return_conditional_losses_50263
XbV
gru_cell_7/add_2:0@__inference_gru_7_layer_call_and_return_conditional_losses_50263
-b+
while/gru_cell_7/beta:0while_body_50167
.b,
while/gru_cell_7/add_2:0while_body_50167
WbU
gru_cell_7/beta:0@__inference_gru_7_layer_call_and_return_conditional_losses_50430
XbV
gru_cell_7/add_2:0@__inference_gru_7_layer_call_and_return_conditional_losses_50430
-b+
while/gru_cell_7/beta:0while_body_50334
.b,
while/gru_cell_7/add_2:0while_body_50334
WbU
gru_cell_7/beta:0@__inference_gru_7_layer_call_and_return_conditional_losses_50597
XbV
gru_cell_7/add_2:0@__inference_gru_7_layer_call_and_return_conditional_losses_50597
-b+
while/gru_cell_7/beta:0while_body_50501
.b,
while/gru_cell_7/add_2:0while_body_50501
QbO
beta:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46492
RbP
add_2:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46492
QbO
beta:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46642
RbP
add_2:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_46642
WbU
gru_cell_8/beta:0@__inference_gru_8_layer_call_and_return_conditional_losses_50808
XbV
gru_cell_8/add_2:0@__inference_gru_8_layer_call_and_return_conditional_losses_50808
-b+
while/gru_cell_8/beta:0while_body_50712
.b,
while/gru_cell_8/add_2:0while_body_50712
WbU
gru_cell_8/beta:0@__inference_gru_8_layer_call_and_return_conditional_losses_50975
XbV
gru_cell_8/add_2:0@__inference_gru_8_layer_call_and_return_conditional_losses_50975
-b+
while/gru_cell_8/beta:0while_body_50879
.b,
while/gru_cell_8/add_2:0while_body_50879
WbU
gru_cell_8/beta:0@__inference_gru_8_layer_call_and_return_conditional_losses_51142
XbV
gru_cell_8/add_2:0@__inference_gru_8_layer_call_and_return_conditional_losses_51142
-b+
while/gru_cell_8/beta:0while_body_51046
.b,
while/gru_cell_8/add_2:0while_body_51046
WbU
gru_cell_8/beta:0@__inference_gru_8_layer_call_and_return_conditional_losses_51309
XbV
gru_cell_8/add_2:0@__inference_gru_8_layer_call_and_return_conditional_losses_51309
-b+
while/gru_cell_8/beta:0while_body_51213
.b,
while/gru_cell_8/add_2:0while_body_51213
QbO
beta:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51402
RbP
add_2:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51402
QbO
beta:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51448
RbP
add_2:0E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51448
QbO
beta:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51522
RbP
add_2:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51522
QbO
beta:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51568
RbP
add_2:0E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51568
QbO
beta:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51642
RbP
add_2:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51642
QbO
beta:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51688
RbP
add_2:0E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51688
 __inference__wrapped_model_45711z867;9:><=)*8¢5
.¢+
)&
gru_6_inputÿÿÿÿÿÿÿÿÿd
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_2_layer_call_and_return_conditional_losses_51328\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_2_layer_call_fn_51318O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿÏ
@__inference_gru_6_layer_call_and_return_conditional_losses_49384867O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 Ï
@__inference_gru_6_layer_call_and_return_conditional_losses_49551867O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 µ
@__inference_gru_6_layer_call_and_return_conditional_losses_49718q867?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿdd
 µ
@__inference_gru_6_layer_call_and_return_conditional_losses_49885q867?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿdd
 ¦
%__inference_gru_6_layer_call_fn_49184}867O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd¦
%__inference_gru_6_layer_call_fn_49195}867O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
%__inference_gru_6_layer_call_fn_49206d867?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿdd
%__inference_gru_6_layer_call_fn_49217d867?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿddÏ
@__inference_gru_7_layer_call_and_return_conditional_losses_50096;9:O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 Ï
@__inference_gru_7_layer_call_and_return_conditional_losses_50263;9:O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 µ
@__inference_gru_7_layer_call_and_return_conditional_losses_50430q;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿdd
 µ
@__inference_gru_7_layer_call_and_return_conditional_losses_50597q;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿdd
 ¦
%__inference_gru_7_layer_call_fn_49896};9:O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd¦
%__inference_gru_7_layer_call_fn_49907};9:O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
%__inference_gru_7_layer_call_fn_49918d;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿdd
%__inference_gru_7_layer_call_fn_49929d;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿddÁ
@__inference_gru_8_layer_call_and_return_conditional_losses_50808}><=O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Á
@__inference_gru_8_layer_call_and_return_conditional_losses_50975}><=O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ±
@__inference_gru_8_layer_call_and_return_conditional_losses_51142m><=?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ±
@__inference_gru_8_layer_call_and_return_conditional_losses_51309m><=?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
%__inference_gru_8_layer_call_fn_50608p><=O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
%__inference_gru_8_layer_call_fn_50619p><=O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
%__inference_gru_8_layer_call_fn_50630`><=?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
%__inference_gru_8_layer_call_fn_50641`><=?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51402·867\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿd
$!

0/1/0ÿÿÿÿÿÿÿÿÿd
 
E__inference_gru_cell_6_layer_call_and_return_conditional_losses_51448·867\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿd
$!

0/1/0ÿÿÿÿÿÿÿÿÿd
 Ø
*__inference_gru_cell_6_layer_call_fn_51342©867\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿd
"

1/0ÿÿÿÿÿÿÿÿÿdØ
*__inference_gru_cell_6_layer_call_fn_51356©867\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿd
"

1/0ÿÿÿÿÿÿÿÿÿd
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51522·;9:\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿd
$!

0/1/0ÿÿÿÿÿÿÿÿÿd
 
E__inference_gru_cell_7_layer_call_and_return_conditional_losses_51568·;9:\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿd
$!

0/1/0ÿÿÿÿÿÿÿÿÿd
 Ø
*__inference_gru_cell_7_layer_call_fn_51462©;9:\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿd
"

1/0ÿÿÿÿÿÿÿÿÿdØ
*__inference_gru_cell_7_layer_call_fn_51476©;9:\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿd
"

1/0ÿÿÿÿÿÿÿÿÿd
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51642·><=\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿd
$!

0/1/0ÿÿÿÿÿÿÿÿÿd
 
E__inference_gru_cell_8_layer_call_and_return_conditional_losses_51688·><=\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿd
$!

0/1/0ÿÿÿÿÿÿÿÿÿd
 Ø
*__inference_gru_cell_8_layer_call_fn_51582©><=\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿd
"

1/0ÿÿÿÿÿÿÿÿÿdØ
*__inference_gru_cell_8_layer_call_fn_51596©><=\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿd
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿd
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿd
"

1/0ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51790e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51808e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51826 e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51844¡¢e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51862£¤e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51880¥¦e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51898§¨e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51916©ªe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51934«¬e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51952­®e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51970¯°e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_51988±²e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52006³´e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52024µ¶e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52042·¸e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52060¹ºe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52078»¼e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52096½¾e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52114¿Àe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52132ÁÂe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52150ÃÄe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52168ÅÆe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52186ÇÈe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52204ÉÊe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52222ËÌe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52240ÍÎe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52258ÏÐe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52276ÑÒe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52294ÓÔe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52312ÕÖe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52330×Øe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52348ÙÚe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52366ÛÜe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52384ÝÞe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52402ßàe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52420áâe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52438ãäe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52456åæe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52474çèe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52492éêe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52510ëìe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52528íîe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52546ïðe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52564ñòe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52582óôe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52600õöe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52618÷øe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52636ùúe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52654ûüe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52672ýþe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52690ÿe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52708e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52726e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52744e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52762e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52780e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52798e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52816e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52834e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52852e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52870e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52888e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52906e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52924e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52942e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdº
"__inference_internal_grad_fn_52960e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿd
(%
result_grads_1ÿÿÿÿÿÿÿÿÿd
ª "$!

 

1ÿÿÿÿÿÿÿÿÿdÁ
G__inference_sequential_2_layer_call_and_return_conditional_losses_48056v867;9:><=)*@¢=
6¢3
)&
gru_6_inputÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
G__inference_sequential_2_layer_call_and_return_conditional_losses_48086v867;9:><=)*@¢=
6¢3
)&
gru_6_inputÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_2_layer_call_and_return_conditional_losses_48645q867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_2_layer_call_and_return_conditional_losses_49144q867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_2_layer_call_fn_47338i867;9:><=)*@¢=
6¢3
)&
gru_6_inputÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_2_layer_call_fn_48026i867;9:><=)*@¢=
6¢3
)&
gru_6_inputÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_2_layer_call_fn_48119d867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_2_layer_call_fn_48146d867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ±
#__inference_signature_wrapper_49173867;9:><=)*G¢D
¢ 
=ª:
8
gru_6_input)&
gru_6_inputÿÿÿÿÿÿÿÿÿd"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ