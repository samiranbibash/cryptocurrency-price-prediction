ü£:
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
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ò7
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:d*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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
gru_3/gru_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*(
shared_namegru_3/gru_cell_9/kernel

+gru_3/gru_cell_9/kernel/Read/ReadVariableOpReadVariableOpgru_3/gru_cell_9/kernel*
_output_shapes
:	¬*
dtype0

!gru_3/gru_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*2
shared_name#!gru_3/gru_cell_9/recurrent_kernel

5gru_3/gru_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_3/gru_cell_9/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_3/gru_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*&
shared_namegru_3/gru_cell_9/bias

)gru_3/gru_cell_9/bias/Read/ReadVariableOpReadVariableOpgru_3/gru_cell_9/bias*
_output_shapes
:	¬*
dtype0

gru_4/gru_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*)
shared_namegru_4/gru_cell_10/kernel

,gru_4/gru_cell_10/kernel/Read/ReadVariableOpReadVariableOpgru_4/gru_cell_10/kernel*
_output_shapes
:	d¬*
dtype0
¡
"gru_4/gru_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*3
shared_name$"gru_4/gru_cell_10/recurrent_kernel

6gru_4/gru_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru_4/gru_cell_10/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_4/gru_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*'
shared_namegru_4/gru_cell_10/bias

*gru_4/gru_cell_10/bias/Read/ReadVariableOpReadVariableOpgru_4/gru_cell_10/bias*
_output_shapes
:	¬*
dtype0

gru_5/gru_cell_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*)
shared_namegru_5/gru_cell_11/kernel

,gru_5/gru_cell_11/kernel/Read/ReadVariableOpReadVariableOpgru_5/gru_cell_11/kernel*
_output_shapes
:	d¬*
dtype0
¡
"gru_5/gru_cell_11/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*3
shared_name$"gru_5/gru_cell_11/recurrent_kernel

6gru_5/gru_cell_11/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru_5/gru_cell_11/recurrent_kernel*
_output_shapes
:	d¬*
dtype0

gru_5/gru_cell_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*'
shared_namegru_5/gru_cell_11/bias

*gru_5/gru_cell_11/bias/Read/ReadVariableOpReadVariableOpgru_5/gru_cell_11/bias*
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
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0

Adam/gru_3/gru_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*/
shared_name Adam/gru_3/gru_cell_9/kernel/m

2Adam/gru_3/gru_cell_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_3/gru_cell_9/kernel/m*
_output_shapes
:	¬*
dtype0
­
(Adam/gru_3/gru_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_3/gru_cell_9/recurrent_kernel/m
¦
<Adam/gru_3/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_3/gru_cell_9/recurrent_kernel/m*
_output_shapes
:	d¬*
dtype0

Adam/gru_3/gru_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_3/gru_cell_9/bias/m

0Adam/gru_3/gru_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_3/gru_cell_9/bias/m*
_output_shapes
:	¬*
dtype0

Adam/gru_4/gru_cell_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*0
shared_name!Adam/gru_4/gru_cell_10/kernel/m

3Adam/gru_4/gru_cell_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_10/kernel/m*
_output_shapes
:	d¬*
dtype0
¯
)Adam/gru_4/gru_cell_10/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*:
shared_name+)Adam/gru_4/gru_cell_10/recurrent_kernel/m
¨
=Adam/gru_4/gru_cell_10/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Adam/gru_4/gru_cell_10/recurrent_kernel/m*
_output_shapes
:	d¬*
dtype0

Adam/gru_4/gru_cell_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*.
shared_nameAdam/gru_4/gru_cell_10/bias/m

1Adam/gru_4/gru_cell_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_10/bias/m*
_output_shapes
:	¬*
dtype0

Adam/gru_5/gru_cell_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*0
shared_name!Adam/gru_5/gru_cell_11/kernel/m

3Adam/gru_5/gru_cell_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_5/gru_cell_11/kernel/m*
_output_shapes
:	d¬*
dtype0
¯
)Adam/gru_5/gru_cell_11/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*:
shared_name+)Adam/gru_5/gru_cell_11/recurrent_kernel/m
¨
=Adam/gru_5/gru_cell_11/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp)Adam/gru_5/gru_cell_11/recurrent_kernel/m*
_output_shapes
:	d¬*
dtype0

Adam/gru_5/gru_cell_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*.
shared_nameAdam/gru_5/gru_cell_11/bias/m

1Adam/gru_5/gru_cell_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_5/gru_cell_11/bias/m*
_output_shapes
:	¬*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0

Adam/gru_3/gru_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*/
shared_name Adam/gru_3/gru_cell_9/kernel/v

2Adam/gru_3/gru_cell_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_3/gru_cell_9/kernel/v*
_output_shapes
:	¬*
dtype0
­
(Adam/gru_3/gru_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*9
shared_name*(Adam/gru_3/gru_cell_9/recurrent_kernel/v
¦
<Adam/gru_3/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_3/gru_cell_9/recurrent_kernel/v*
_output_shapes
:	d¬*
dtype0

Adam/gru_3/gru_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*-
shared_nameAdam/gru_3/gru_cell_9/bias/v

0Adam/gru_3/gru_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_3/gru_cell_9/bias/v*
_output_shapes
:	¬*
dtype0

Adam/gru_4/gru_cell_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*0
shared_name!Adam/gru_4/gru_cell_10/kernel/v

3Adam/gru_4/gru_cell_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_10/kernel/v*
_output_shapes
:	d¬*
dtype0
¯
)Adam/gru_4/gru_cell_10/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*:
shared_name+)Adam/gru_4/gru_cell_10/recurrent_kernel/v
¨
=Adam/gru_4/gru_cell_10/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Adam/gru_4/gru_cell_10/recurrent_kernel/v*
_output_shapes
:	d¬*
dtype0

Adam/gru_4/gru_cell_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*.
shared_nameAdam/gru_4/gru_cell_10/bias/v

1Adam/gru_4/gru_cell_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_4/gru_cell_10/bias/v*
_output_shapes
:	¬*
dtype0

Adam/gru_5/gru_cell_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*0
shared_name!Adam/gru_5/gru_cell_11/kernel/v

3Adam/gru_5/gru_cell_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_5/gru_cell_11/kernel/v*
_output_shapes
:	d¬*
dtype0
¯
)Adam/gru_5/gru_cell_11/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d¬*:
shared_name+)Adam/gru_5/gru_cell_11/recurrent_kernel/v
¨
=Adam/gru_5/gru_cell_11/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp)Adam/gru_5/gru_cell_11/recurrent_kernel/v*
_output_shapes
:	d¬*
dtype0

Adam/gru_5/gru_cell_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*.
shared_nameAdam/gru_5/gru_cell_11/bias/v

1Adam/gru_5/gru_cell_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_5/gru_cell_11/bias/v*
_output_shapes
:	¬*
dtype0

NoOpNoOp
¨M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ãL
valueÙLBÖL BÏL
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
VARIABLE_VALUEdense_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEgru_3/gru_cell_9/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_3/gru_cell_9/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_3/gru_cell_9/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEgru_4/gru_cell_10/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"gru_4/gru_cell_10/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEgru_4/gru_cell_10/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEgru_5/gru_cell_11/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"gru_5/gru_cell_11/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEgru_5/gru_cell_11/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_3/gru_cell_9/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_3/gru_cell_9/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_3/gru_cell_9/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_4/gru_cell_10/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/gru_4/gru_cell_10/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_4/gru_cell_10/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_5/gru_cell_11/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/gru_5/gru_cell_11/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_5/gru_cell_11/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_3/gru_cell_9/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_3/gru_cell_9/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_3/gru_cell_9/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_4/gru_cell_10/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/gru_4/gru_cell_10/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_4/gru_cell_10/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/gru_5/gru_cell_11/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/gru_5/gru_cell_11/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gru_5/gru_cell_11/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_gru_3_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿd
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_3_inputgru_3/gru_cell_9/biasgru_3/gru_cell_9/kernel!gru_3/gru_cell_9/recurrent_kernelgru_4/gru_cell_10/biasgru_4/gru_cell_10/kernel"gru_4/gru_cell_10/recurrent_kernelgru_5/gru_cell_11/biasgru_5/gru_cell_11/kernel"gru_5/gru_cell_11/recurrent_kerneldense_6/kerneldense_6/bias*
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
GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_140526
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+gru_3/gru_cell_9/kernel/Read/ReadVariableOp5gru_3/gru_cell_9/recurrent_kernel/Read/ReadVariableOp)gru_3/gru_cell_9/bias/Read/ReadVariableOp,gru_4/gru_cell_10/kernel/Read/ReadVariableOp6gru_4/gru_cell_10/recurrent_kernel/Read/ReadVariableOp*gru_4/gru_cell_10/bias/Read/ReadVariableOp,gru_5/gru_cell_11/kernel/Read/ReadVariableOp6gru_5/gru_cell_11/recurrent_kernel/Read/ReadVariableOp*gru_5/gru_cell_11/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp2Adam/gru_3/gru_cell_9/kernel/m/Read/ReadVariableOp<Adam/gru_3/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_3/gru_cell_9/bias/m/Read/ReadVariableOp3Adam/gru_4/gru_cell_10/kernel/m/Read/ReadVariableOp=Adam/gru_4/gru_cell_10/recurrent_kernel/m/Read/ReadVariableOp1Adam/gru_4/gru_cell_10/bias/m/Read/ReadVariableOp3Adam/gru_5/gru_cell_11/kernel/m/Read/ReadVariableOp=Adam/gru_5/gru_cell_11/recurrent_kernel/m/Read/ReadVariableOp1Adam/gru_5/gru_cell_11/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp2Adam/gru_3/gru_cell_9/kernel/v/Read/ReadVariableOp<Adam/gru_3/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_3/gru_cell_9/bias/v/Read/ReadVariableOp3Adam/gru_4/gru_cell_10/kernel/v/Read/ReadVariableOp=Adam/gru_4/gru_cell_10/recurrent_kernel/v/Read/ReadVariableOp1Adam/gru_4/gru_cell_10/bias/v/Read/ReadVariableOp3Adam/gru_5/gru_cell_11/kernel/v/Read/ReadVariableOp=Adam/gru_5/gru_cell_11/recurrent_kernel/v/Read/ReadVariableOp1Adam/gru_5/gru_cell_11/bias/v/Read/ReadVariableOpConst*5
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
GPU 2J 8 *(
f#R!
__inference__traced_save_144372

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_3/gru_cell_9/kernel!gru_3/gru_cell_9/recurrent_kernelgru_3/gru_cell_9/biasgru_4/gru_cell_10/kernel"gru_4/gru_cell_10/recurrent_kernelgru_4/gru_cell_10/biasgru_5/gru_cell_11/kernel"gru_5/gru_cell_11/recurrent_kernelgru_5/gru_cell_11/biastotalcountAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/gru_3/gru_cell_9/kernel/m(Adam/gru_3/gru_cell_9/recurrent_kernel/mAdam/gru_3/gru_cell_9/bias/mAdam/gru_4/gru_cell_10/kernel/m)Adam/gru_4/gru_cell_10/recurrent_kernel/mAdam/gru_4/gru_cell_10/bias/mAdam/gru_5/gru_cell_11/kernel/m)Adam/gru_5/gru_cell_11/recurrent_kernel/mAdam/gru_5/gru_cell_11/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/gru_3/gru_cell_9/kernel/v(Adam/gru_3/gru_cell_9/recurrent_kernel/vAdam/gru_3/gru_cell_9/bias/vAdam/gru_4/gru_cell_10/kernel/v)Adam/gru_4/gru_cell_10/recurrent_kernel/vAdam/gru_4/gru_cell_10/bias/vAdam/gru_5/gru_cell_11/kernel/v)Adam/gru_5/gru_cell_11/recurrent_kernel/vAdam/gru_5/gru_cell_11/bias/v*4
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_144502»È5


#__inference_internal_grad_fn_144133
result_grads_0
result_grads_1
mul_while_gru_cell_11_beta
mul_while_gru_cell_11_add_2
identity
mulMulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2*
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
4
ÿ
A__inference_gru_3_layer_call_and_return_conditional_losses_137407

inputs$
gru_cell_9_137331:	¬$
gru_cell_9_137333:	¬$
gru_cell_9_137335:	d¬
identity¢"gru_cell_9/StatefulPartitionedCall¢while;
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
shrink_axis_maskÄ
"gru_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9_137331gru_cell_9_137333gru_cell_9_137335*
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
GPU 2J 8 *O
fJRH
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137291n
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
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9_137331gru_cell_9_137333gru_cell_9_137335*
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
bodyR
while_body_137343*
condR
while_cond_137342*8
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
NoOpNoOp#^gru_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"gru_cell_9/StatefulPartitionedCall"gru_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷B

while_body_142065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_11_readvariableop_resource_0:	¬E
2while_gru_cell_11_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_11_readvariableop_resource:	¬C
0while_gru_cell_11_matmul_readvariableop_resource:	d¬E
2while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_11/MatMul/ReadVariableOp¢)while/gru_cell_11/MatMul_1/ReadVariableOp¢ while/gru_cell_11/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_11/ReadVariableOpReadVariableOp+while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_11/unstackUnpack(while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_11/BiasAddBiasAdd"while/gru_cell_11/MatMul:product:0"while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_11/splitSplit*while/gru_cell_11/split/split_dim:output:0"while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_11/MatMul_1MatMulwhile_placeholder_21while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_11/BiasAdd_1BiasAdd$while/gru_cell_11/MatMul_1:product:0"while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_11/split_1SplitV$while/gru_cell_11/BiasAdd_1:output:0 while/gru_cell_11/Const:output:0,while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_11/addAddV2 while/gru_cell_11/split:output:0"while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_11/SigmoidSigmoidwhile/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_1AddV2 while/gru_cell_11/split:output:1"while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_1Sigmoidwhile/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mulMulwhile/gru_cell_11/Sigmoid_1:y:0"while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_2AddV2 while/gru_cell_11/split:output:2while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/mul_1Mulwhile/gru_cell_11/beta:output:0while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_2Sigmoidwhile/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_2Mulwhile/gru_cell_11/add_2:z:0while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/IdentityIdentitywhile/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_11/IdentityN	IdentityNwhile/gru_cell_11/mul_2:z:0while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142115*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_3Mulwhile/gru_cell_11/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/subSub while/gru_cell_11/sub/x:output:0while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_4Mulwhile/gru_cell_11/sub:z:0$while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_3AddV2while/gru_cell_11/mul_3:z:0while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_11/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_11/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_11/MatMul/ReadVariableOp*^while/gru_cell_11/MatMul_1/ReadVariableOp!^while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_11_matmul_1_readvariableop_resource4while_gru_cell_11_matmul_1_readvariableop_resource_0"f
0while_gru_cell_11_matmul_readvariableop_resource2while_gru_cell_11_matmul_readvariableop_resource_0"X
)while_gru_cell_11_readvariableop_resource+while_gru_cell_11_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_11/MatMul/ReadVariableOp'while/gru_cell_11/MatMul/ReadVariableOp2V
)while/gru_cell_11/MatMul_1/ReadVariableOp)while/gru_cell_11/MatMul_1/ReadVariableOp2D
 while/gru_cell_11/ReadVariableOp while/gru_cell_11/ReadVariableOp: 
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
4

A__inference_gru_4_layer_call_and_return_conditional_losses_137759

inputs%
gru_cell_10_137683:	¬%
gru_cell_10_137685:	d¬%
gru_cell_10_137687:	d¬
identity¢#gru_cell_10/StatefulPartitionedCall¢while;
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
shrink_axis_maskÉ
#gru_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_10_137683gru_cell_10_137685gru_cell_10_137687*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137643n
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
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_10_137683gru_cell_10_137685gru_cell_10_137687*
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
bodyR
while_body_137695*
condR
while_cond_137694*8
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdt
NoOpNoOp$^gru_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2J
#gru_cell_10/StatefulPartitionedCall#gru_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ÝR

A__inference_gru_5_layer_call_and_return_conditional_losses_142328
inputs_06
#gru_cell_11_readvariableop_resource:	¬=
*gru_cell_11_matmul_readvariableop_resource:	d¬?
,gru_cell_11_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_11/MatMul/ReadVariableOp¢#gru_cell_11/MatMul_1/ReadVariableOp¢gru_cell_11/ReadVariableOp¢while=
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
shrink_axis_mask
gru_cell_11/ReadVariableOpReadVariableOp#gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_11/unstackUnpack"gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_11/MatMul/ReadVariableOpReadVariableOp*gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMulMatMulstrided_slice_2:output:0)gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAddBiasAddgru_cell_11/MatMul:product:0gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_11/splitSplit$gru_cell_11/split/split_dim:output:0gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMul_1MatMulzeros:output:0+gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAdd_1BiasAddgru_cell_11/MatMul_1:product:0gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_11/split_1SplitVgru_cell_11/BiasAdd_1:output:0gru_cell_11/Const:output:0&gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_11/addAddV2gru_cell_11/split:output:0gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_11/SigmoidSigmoidgru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/add_1AddV2gru_cell_11/split:output:1gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_1Sigmoidgru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mulMulgru_cell_11/Sigmoid_1:y:0gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_11/add_2AddV2gru_cell_11/split:output:2gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_11/mul_1Mulgru_cell_11/beta:output:0gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_2Sigmoidgru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_11/mul_2Mulgru_cell_11/add_2:z:0gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/IdentityIdentitygru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_11/IdentityN	IdentityNgru_cell_11/mul_2:z:0gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142216*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_11/mul_3Mulgru_cell_11/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_11/subSubgru_cell_11/sub/x:output:0gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mul_4Mulgru_cell_11/sub:z:0gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_11/add_3AddV2gru_cell_11/mul_3:z:0gru_cell_11/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_11_readvariableop_resource*gru_cell_11_matmul_readvariableop_resource,gru_cell_11_matmul_1_readvariableop_resource*
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
bodyR
while_body_142232*
condR
while_cond_142231*8
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
:ÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_11/MatMul/ReadVariableOp$^gru_cell_11/MatMul_1/ReadVariableOp^gru_cell_11/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2F
!gru_cell_11/MatMul/ReadVariableOp!gru_cell_11/MatMul/ReadVariableOp2J
#gru_cell_11/MatMul_1/ReadVariableOp#gru_cell_11/MatMul_1/ReadVariableOp28
gru_cell_11/ReadVariableOpgru_cell_11/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
!
Ü
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142755

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142741*:
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
Ú
ª
while_cond_141519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_141519___redundant_placeholder04
0while_while_cond_141519___redundant_placeholder14
0while_while_cond_141519___redundant_placeholder24
0while_while_cond_141519___redundant_placeholder3
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
ç

¨
-__inference_sequential_6_layer_call_fn_139472

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
identity¢StatefulPartitionedCallÒ
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_138666o
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
Ó
¶
#__inference_internal_grad_fn_143161
result_grads_0
result_grads_1+
'mul_sequential_6_gru_4_gru_cell_10_beta,
(mul_sequential_6_gru_4_gru_cell_10_add_2
identity 
mulMul'mul_sequential_6_gru_4_gru_cell_10_beta(mul_sequential_6_gru_4_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul'mul_sequential_6_gru_4_gru_cell_10_beta(mul_sequential_6_gru_4_gru_cell_10_add_2*
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


#__inference_internal_grad_fn_143305
result_grads_0
result_grads_1
mul_while_gru_cell_10_beta
mul_while_gru_cell_10_add_2
identity
mulMulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2*
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


#__inference_internal_grad_fn_143449
result_grads_0
result_grads_1
mul_while_gru_cell_11_beta
mul_while_gru_cell_11_add_2
identity
mulMulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2*
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
÷B

while_body_142232
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_11_readvariableop_resource_0:	¬E
2while_gru_cell_11_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_11_readvariableop_resource:	¬C
0while_gru_cell_11_matmul_readvariableop_resource:	d¬E
2while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_11/MatMul/ReadVariableOp¢)while/gru_cell_11/MatMul_1/ReadVariableOp¢ while/gru_cell_11/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_11/ReadVariableOpReadVariableOp+while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_11/unstackUnpack(while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_11/BiasAddBiasAdd"while/gru_cell_11/MatMul:product:0"while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_11/splitSplit*while/gru_cell_11/split/split_dim:output:0"while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_11/MatMul_1MatMulwhile_placeholder_21while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_11/BiasAdd_1BiasAdd$while/gru_cell_11/MatMul_1:product:0"while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_11/split_1SplitV$while/gru_cell_11/BiasAdd_1:output:0 while/gru_cell_11/Const:output:0,while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_11/addAddV2 while/gru_cell_11/split:output:0"while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_11/SigmoidSigmoidwhile/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_1AddV2 while/gru_cell_11/split:output:1"while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_1Sigmoidwhile/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mulMulwhile/gru_cell_11/Sigmoid_1:y:0"while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_2AddV2 while/gru_cell_11/split:output:2while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/mul_1Mulwhile/gru_cell_11/beta:output:0while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_2Sigmoidwhile/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_2Mulwhile/gru_cell_11/add_2:z:0while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/IdentityIdentitywhile/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_11/IdentityN	IdentityNwhile/gru_cell_11/mul_2:z:0while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142282*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_3Mulwhile/gru_cell_11/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/subSub while/gru_cell_11/sub/x:output:0while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_4Mulwhile/gru_cell_11/sub:z:0$while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_3AddV2while/gru_cell_11/mul_3:z:0while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_11/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_11/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_11/MatMul/ReadVariableOp*^while/gru_cell_11/MatMul_1/ReadVariableOp!^while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_11_matmul_1_readvariableop_resource4while_gru_cell_11_matmul_1_readvariableop_resource_0"f
0while_gru_cell_11_matmul_readvariableop_resource2while_gru_cell_11_matmul_readvariableop_resource_0"X
)while_gru_cell_11_readvariableop_resource+while_gru_cell_11_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_11/MatMul/ReadVariableOp'while/gru_cell_11/MatMul/ReadVariableOp2V
)while/gru_cell_11/MatMul_1/ReadVariableOp)while/gru_cell_11/MatMul_1/ReadVariableOp2D
 while/gru_cell_11/ReadVariableOp while/gru_cell_11/ReadVariableOp: 
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
ß

#__inference_internal_grad_fn_143431
result_grads_0
result_grads_1
mul_gru_cell_11_beta
mul_gru_cell_11_add_2
identityz
mulMulmul_gru_cell_11_betamul_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_11_betamul_gru_cell_11_add_2*
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
ß

#__inference_internal_grad_fn_143323
result_grads_0
result_grads_1
mul_gru_cell_11_beta
mul_gru_cell_11_add_2
identityz
mulMulmul_gru_cell_11_betamul_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_11_betamul_gru_cell_11_add_2*
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
V

__inference__traced_save_144372
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_gru_3_gru_cell_9_kernel_read_readvariableop@
<savev2_gru_3_gru_cell_9_recurrent_kernel_read_readvariableop4
0savev2_gru_3_gru_cell_9_bias_read_readvariableop7
3savev2_gru_4_gru_cell_10_kernel_read_readvariableopA
=savev2_gru_4_gru_cell_10_recurrent_kernel_read_readvariableop5
1savev2_gru_4_gru_cell_10_bias_read_readvariableop7
3savev2_gru_5_gru_cell_11_kernel_read_readvariableopA
=savev2_gru_5_gru_cell_11_recurrent_kernel_read_readvariableop5
1savev2_gru_5_gru_cell_11_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop=
9savev2_adam_gru_3_gru_cell_9_kernel_m_read_readvariableopG
Csavev2_adam_gru_3_gru_cell_9_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_3_gru_cell_9_bias_m_read_readvariableop>
:savev2_adam_gru_4_gru_cell_10_kernel_m_read_readvariableopH
Dsavev2_adam_gru_4_gru_cell_10_recurrent_kernel_m_read_readvariableop<
8savev2_adam_gru_4_gru_cell_10_bias_m_read_readvariableop>
:savev2_adam_gru_5_gru_cell_11_kernel_m_read_readvariableopH
Dsavev2_adam_gru_5_gru_cell_11_recurrent_kernel_m_read_readvariableop<
8savev2_adam_gru_5_gru_cell_11_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop=
9savev2_adam_gru_3_gru_cell_9_kernel_v_read_readvariableopG
Csavev2_adam_gru_3_gru_cell_9_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_3_gru_cell_9_bias_v_read_readvariableop>
:savev2_adam_gru_4_gru_cell_10_kernel_v_read_readvariableopH
Dsavev2_adam_gru_4_gru_cell_10_recurrent_kernel_v_read_readvariableop<
8savev2_adam_gru_4_gru_cell_10_bias_v_read_readvariableop>
:savev2_adam_gru_5_gru_cell_11_kernel_v_read_readvariableopH
Dsavev2_adam_gru_5_gru_cell_11_recurrent_kernel_v_read_readvariableop<
8savev2_adam_gru_5_gru_cell_11_bias_v_read_readvariableop
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
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ô
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_gru_3_gru_cell_9_kernel_read_readvariableop<savev2_gru_3_gru_cell_9_recurrent_kernel_read_readvariableop0savev2_gru_3_gru_cell_9_bias_read_readvariableop3savev2_gru_4_gru_cell_10_kernel_read_readvariableop=savev2_gru_4_gru_cell_10_recurrent_kernel_read_readvariableop1savev2_gru_4_gru_cell_10_bias_read_readvariableop3savev2_gru_5_gru_cell_11_kernel_read_readvariableop=savev2_gru_5_gru_cell_11_recurrent_kernel_read_readvariableop1savev2_gru_5_gru_cell_11_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop9savev2_adam_gru_3_gru_cell_9_kernel_m_read_readvariableopCsavev2_adam_gru_3_gru_cell_9_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_3_gru_cell_9_bias_m_read_readvariableop:savev2_adam_gru_4_gru_cell_10_kernel_m_read_readvariableopDsavev2_adam_gru_4_gru_cell_10_recurrent_kernel_m_read_readvariableop8savev2_adam_gru_4_gru_cell_10_bias_m_read_readvariableop:savev2_adam_gru_5_gru_cell_11_kernel_m_read_readvariableopDsavev2_adam_gru_5_gru_cell_11_recurrent_kernel_m_read_readvariableop8savev2_adam_gru_5_gru_cell_11_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop9savev2_adam_gru_3_gru_cell_9_kernel_v_read_readvariableopCsavev2_adam_gru_3_gru_cell_9_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_3_gru_cell_9_bias_v_read_readvariableop:savev2_adam_gru_4_gru_cell_10_kernel_v_read_readvariableopDsavev2_adam_gru_4_gru_cell_10_recurrent_kernel_v_read_readvariableop8savev2_adam_gru_4_gru_cell_10_bias_v_read_readvariableop:savev2_adam_gru_5_gru_cell_11_kernel_v_read_readvariableopDsavev2_adam_gru_5_gru_cell_11_recurrent_kernel_v_read_readvariableop8savev2_adam_gru_5_gru_cell_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
£R

A__inference_gru_4_layer_call_and_return_conditional_losses_141783

inputs6
#gru_cell_10_readvariableop_resource:	¬=
*gru_cell_10_matmul_readvariableop_resource:	d¬?
,gru_cell_10_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_10/MatMul/ReadVariableOp¢#gru_cell_10/MatMul_1/ReadVariableOp¢gru_cell_10/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_10/ReadVariableOpReadVariableOp#gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_10/unstackUnpack"gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_10/MatMul/ReadVariableOpReadVariableOp*gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMulMatMulstrided_slice_2:output:0)gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAddBiasAddgru_cell_10/MatMul:product:0gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_10/splitSplit$gru_cell_10/split/split_dim:output:0gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMul_1MatMulzeros:output:0+gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAdd_1BiasAddgru_cell_10/MatMul_1:product:0gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_10/split_1SplitVgru_cell_10/BiasAdd_1:output:0gru_cell_10/Const:output:0&gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_10/addAddV2gru_cell_10/split:output:0gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_10/SigmoidSigmoidgru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/add_1AddV2gru_cell_10/split:output:1gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_1Sigmoidgru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mulMulgru_cell_10/Sigmoid_1:y:0gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_10/add_2AddV2gru_cell_10/split:output:2gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_10/mul_1Mulgru_cell_10/beta:output:0gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_2Sigmoidgru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_10/mul_2Mulgru_cell_10/add_2:z:0gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/IdentityIdentitygru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_10/IdentityN	IdentityNgru_cell_10/mul_2:z:0gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141671*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_10/mul_3Mulgru_cell_10/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_10/subSubgru_cell_10/sub/x:output:0gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mul_4Mulgru_cell_10/sub:z:0gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_10/add_3AddV2gru_cell_10/mul_3:z:0gru_cell_10/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_10_readvariableop_resource*gru_cell_10_matmul_readvariableop_resource,gru_cell_10_matmul_1_readvariableop_resource*
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
bodyR
while_body_141687*
condR
while_cond_141686*8
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
:ÿÿÿÿÿÿÿÿÿddµ
NoOpNoOp"^gru_cell_10/MatMul/ReadVariableOp$^gru_cell_10/MatMul_1/ReadVariableOp^gru_cell_10/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_10/MatMul/ReadVariableOp!gru_cell_10/MatMul/ReadVariableOp2J
#gru_cell_10/MatMul_1/ReadVariableOp#gru_cell_10/MatMul_1/ReadVariableOp28
gru_cell_10/ReadVariableOpgru_cell_10/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs

x
#__inference_internal_grad_fn_144277
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
÷B

while_body_142399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_11_readvariableop_resource_0:	¬E
2while_gru_cell_11_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_11_readvariableop_resource:	¬C
0while_gru_cell_11_matmul_readvariableop_resource:	d¬E
2while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_11/MatMul/ReadVariableOp¢)while/gru_cell_11/MatMul_1/ReadVariableOp¢ while/gru_cell_11/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_11/ReadVariableOpReadVariableOp+while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_11/unstackUnpack(while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_11/BiasAddBiasAdd"while/gru_cell_11/MatMul:product:0"while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_11/splitSplit*while/gru_cell_11/split/split_dim:output:0"while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_11/MatMul_1MatMulwhile_placeholder_21while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_11/BiasAdd_1BiasAdd$while/gru_cell_11/MatMul_1:product:0"while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_11/split_1SplitV$while/gru_cell_11/BiasAdd_1:output:0 while/gru_cell_11/Const:output:0,while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_11/addAddV2 while/gru_cell_11/split:output:0"while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_11/SigmoidSigmoidwhile/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_1AddV2 while/gru_cell_11/split:output:1"while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_1Sigmoidwhile/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mulMulwhile/gru_cell_11/Sigmoid_1:y:0"while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_2AddV2 while/gru_cell_11/split:output:2while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/mul_1Mulwhile/gru_cell_11/beta:output:0while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_2Sigmoidwhile/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_2Mulwhile/gru_cell_11/add_2:z:0while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/IdentityIdentitywhile/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_11/IdentityN	IdentityNwhile/gru_cell_11/mul_2:z:0while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142449*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_3Mulwhile/gru_cell_11/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/subSub while/gru_cell_11/sub/x:output:0while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_4Mulwhile/gru_cell_11/sub:z:0$while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_3AddV2while/gru_cell_11/mul_3:z:0while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_11/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_11/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_11/MatMul/ReadVariableOp*^while/gru_cell_11/MatMul_1/ReadVariableOp!^while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_11_matmul_1_readvariableop_resource4while_gru_cell_11_matmul_1_readvariableop_resource_0"f
0while_gru_cell_11_matmul_readvariableop_resource2while_gru_cell_11_matmul_readvariableop_resource_0"X
)while_gru_cell_11_readvariableop_resource+while_gru_cell_11_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_11/MatMul/ReadVariableOp'while/gru_cell_11/MatMul/ReadVariableOp2V
)while/gru_cell_11/MatMul_1/ReadVariableOp)while/gru_cell_11/MatMul_1/ReadVariableOp2D
 while/gru_cell_11/ReadVariableOp while/gru_cell_11/ReadVariableOp: 
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
 
µ
while_body_137695
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_10_137717_0:	¬-
while_gru_cell_10_137719_0:	d¬-
while_gru_cell_10_137721_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_10_137717:	¬+
while_gru_cell_10_137719:	d¬+
while_gru_cell_10_137721:	d¬¢)while/gru_cell_10/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
)while/gru_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_10_137717_0while_gru_cell_10_137719_0while_gru_cell_10_137721_0*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137643Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_10/StatefulPartitionedCall:output:0*
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
: :éèÒ
while/Identity_4Identity2while/gru_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx

while/NoOpNoOp*^while/gru_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_10_137717while_gru_cell_10_137717_0"6
while_gru_cell_10_137719while_gru_cell_10_137719_0"6
while_gru_cell_10_137721while_gru_cell_10_137721_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2V
)while/gru_cell_10/StatefulPartitionedCall)while/gru_cell_10/StatefulPartitionedCall: 
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
ç

¨
-__inference_sequential_6_layer_call_fn_139499

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
identity¢StatefulPartitionedCallÒ
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_139327o
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
÷B

while_body_138371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_10_readvariableop_resource_0:	¬E
2while_gru_cell_10_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_10_readvariableop_resource:	¬C
0while_gru_cell_10_matmul_readvariableop_resource:	d¬E
2while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_10/MatMul/ReadVariableOp¢)while/gru_cell_10/MatMul_1/ReadVariableOp¢ while/gru_cell_10/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_10/ReadVariableOpReadVariableOp+while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_10/unstackUnpack(while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_10/BiasAddBiasAdd"while/gru_cell_10/MatMul:product:0"while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_10/splitSplit*while/gru_cell_10/split/split_dim:output:0"while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_10/MatMul_1MatMulwhile_placeholder_21while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_10/BiasAdd_1BiasAdd$while/gru_cell_10/MatMul_1:product:0"while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_10/split_1SplitV$while/gru_cell_10/BiasAdd_1:output:0 while/gru_cell_10/Const:output:0,while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_10/addAddV2 while/gru_cell_10/split:output:0"while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_10/SigmoidSigmoidwhile/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_1AddV2 while/gru_cell_10/split:output:1"while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_1Sigmoidwhile/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mulMulwhile/gru_cell_10/Sigmoid_1:y:0"while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_2AddV2 while/gru_cell_10/split:output:2while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/mul_1Mulwhile/gru_cell_10/beta:output:0while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_2Sigmoidwhile/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_2Mulwhile/gru_cell_10/add_2:z:0while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/IdentityIdentitywhile/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_10/IdentityN	IdentityNwhile/gru_cell_10/mul_2:z:0while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138421*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_3Mulwhile/gru_cell_10/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/subSub while/gru_cell_10/sub/x:output:0while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_4Mulwhile/gru_cell_10/sub:z:0$while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_3AddV2while/gru_cell_10/mul_3:z:0while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_10/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_10/MatMul/ReadVariableOp*^while/gru_cell_10/MatMul_1/ReadVariableOp!^while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_10_matmul_1_readvariableop_resource4while_gru_cell_10_matmul_1_readvariableop_resource_0"f
0while_gru_cell_10_matmul_readvariableop_resource2while_gru_cell_10_matmul_readvariableop_resource_0"X
)while_gru_cell_10_readvariableop_resource+while_gru_cell_10_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_10/MatMul/ReadVariableOp'while/gru_cell_10/MatMul/ReadVariableOp2V
)while/gru_cell_10/MatMul_1/ReadVariableOp)while/gru_cell_10/MatMul_1/ReadVariableOp2D
 while/gru_cell_10/ReadVariableOp while/gru_cell_10/ReadVariableOp: 
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
Ú
ª
while_cond_141853
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_141853___redundant_placeholder04
0while_while_cond_141853___redundant_placeholder14
0while_while_cond_141853___redundant_placeholder24
0while_while_cond_141853___redundant_placeholder3
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
!
Û
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137643

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137629*:
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
!
Û
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137845

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137831*:
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
Ú
ª
while_cond_142231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_142231___redundant_placeholder04
0while_while_cond_142231___redundant_placeholder14
0while_while_cond_142231___redundant_placeholder24
0while_while_cond_142231___redundant_placeholder3
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
Æ	
ô
C__inference_dense_6_layer_call_and_return_conditional_losses_142681

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
÷B

while_body_141353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_10_readvariableop_resource_0:	¬E
2while_gru_cell_10_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_10_readvariableop_resource:	¬C
0while_gru_cell_10_matmul_readvariableop_resource:	d¬E
2while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_10/MatMul/ReadVariableOp¢)while/gru_cell_10/MatMul_1/ReadVariableOp¢ while/gru_cell_10/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_10/ReadVariableOpReadVariableOp+while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_10/unstackUnpack(while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_10/BiasAddBiasAdd"while/gru_cell_10/MatMul:product:0"while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_10/splitSplit*while/gru_cell_10/split/split_dim:output:0"while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_10/MatMul_1MatMulwhile_placeholder_21while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_10/BiasAdd_1BiasAdd$while/gru_cell_10/MatMul_1:product:0"while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_10/split_1SplitV$while/gru_cell_10/BiasAdd_1:output:0 while/gru_cell_10/Const:output:0,while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_10/addAddV2 while/gru_cell_10/split:output:0"while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_10/SigmoidSigmoidwhile/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_1AddV2 while/gru_cell_10/split:output:1"while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_1Sigmoidwhile/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mulMulwhile/gru_cell_10/Sigmoid_1:y:0"while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_2AddV2 while/gru_cell_10/split:output:2while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/mul_1Mulwhile/gru_cell_10/beta:output:0while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_2Sigmoidwhile/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_2Mulwhile/gru_cell_10/add_2:z:0while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/IdentityIdentitywhile/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_10/IdentityN	IdentityNwhile/gru_cell_10/mul_2:z:0while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141403*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_3Mulwhile/gru_cell_10/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/subSub while/gru_cell_10/sub/x:output:0while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_4Mulwhile/gru_cell_10/sub:z:0$while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_3AddV2while/gru_cell_10/mul_3:z:0while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_10/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_10/MatMul/ReadVariableOp*^while/gru_cell_10/MatMul_1/ReadVariableOp!^while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_10_matmul_1_readvariableop_resource4while_gru_cell_10_matmul_1_readvariableop_resource_0"f
0while_gru_cell_10_matmul_readvariableop_resource2while_gru_cell_10_matmul_readvariableop_resource_0"X
)while_gru_cell_10_readvariableop_resource+while_gru_cell_10_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_10/MatMul/ReadVariableOp'while/gru_cell_10/MatMul/ReadVariableOp2V
)while/gru_cell_10/MatMul_1/ReadVariableOp)while/gru_cell_10/MatMul_1/ReadVariableOp2D
 while/gru_cell_10/ReadVariableOp while/gru_cell_10/ReadVariableOp: 
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
!
Ý
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_143041

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-143027*:
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
ß

#__inference_internal_grad_fn_143395
result_grads_0
result_grads_1
mul_gru_cell_10_beta
mul_gru_cell_10_add_2
identityz
mulMulmul_gru_cell_10_betamul_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_10_betamul_gru_cell_10_add_2*
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
Ü

$sequential_6_gru_5_while_cond_136961B
>sequential_6_gru_5_while_sequential_6_gru_5_while_loop_counterH
Dsequential_6_gru_5_while_sequential_6_gru_5_while_maximum_iterations(
$sequential_6_gru_5_while_placeholder*
&sequential_6_gru_5_while_placeholder_1*
&sequential_6_gru_5_while_placeholder_2D
@sequential_6_gru_5_while_less_sequential_6_gru_5_strided_slice_1Z
Vsequential_6_gru_5_while_sequential_6_gru_5_while_cond_136961___redundant_placeholder0Z
Vsequential_6_gru_5_while_sequential_6_gru_5_while_cond_136961___redundant_placeholder1Z
Vsequential_6_gru_5_while_sequential_6_gru_5_while_cond_136961___redundant_placeholder2Z
Vsequential_6_gru_5_while_sequential_6_gru_5_while_cond_136961___redundant_placeholder3%
!sequential_6_gru_5_while_identity
®
sequential_6/gru_5/while/LessLess$sequential_6_gru_5_while_placeholder@sequential_6_gru_5_while_less_sequential_6_gru_5_strided_slice_1*
T0*
_output_shapes
: q
!sequential_6/gru_5/while/IdentityIdentity!sequential_6/gru_5/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_6_gru_5_while_identity*sequential_6/gru_5/while/Identity:output:0*(
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
þ

#__inference_internal_grad_fn_143809
result_grads_0
result_grads_1
mul_while_gru_cell_9_beta
mul_while_gru_cell_9_add_2
identity
mulMulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2*
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
R

A__inference_gru_3_layer_call_and_return_conditional_losses_140737
inputs_05
"gru_cell_9_readvariableop_resource:	¬<
)gru_cell_9_matmul_readvariableop_resource:	¬>
+gru_cell_9_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_9/MatMul/ReadVariableOp¢"gru_cell_9/MatMul_1/ReadVariableOp¢gru_cell_9/ReadVariableOp¢while=
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
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_9/mul_1Mulgru_cell_9/beta:output:0gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_2Sigmoidgru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_9/mul_2Mulgru_cell_9/add_2:z:0gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/IdentityIdentitygru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
gru_cell_9/IdentityN	IdentityNgru_cell_9/mul_2:z:0gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140625*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_9/mul_3Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_9/mul_4Mulgru_cell_9/sub:z:0gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_9/add_3AddV2gru_cell_9/mul_3:z:0gru_cell_9/mul_4:z:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
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
bodyR
while_body_140641*
condR
while_cond_140640*8
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
NoOpNoOp!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
B
ÿ
while_body_139163
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_9_readvariableop_resource_0:	¬D
1while_gru_cell_9_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_9_readvariableop_resource:	¬B
/while_gru_cell_9_matmul_readvariableop_resource:	¬D
1while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_9/MatMul/ReadVariableOp¢(while/gru_cell_9/MatMul_1/ReadVariableOp¢while/gru_cell_9/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0while/gru_cell_9/Const:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/beta:output:0while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_2Sigmoidwhile/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/add_2:z:0while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/IdentityIdentitywhile/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
while/gru_cell_9/IdentityN	IdentityNwhile/gru_cell_9/mul_2:z:0while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139213*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_3Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_4Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_3:z:0while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
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
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 
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
ü

gru_3_while_cond_139569(
$gru_3_while_gru_3_while_loop_counter.
*gru_3_while_gru_3_while_maximum_iterations
gru_3_while_placeholder
gru_3_while_placeholder_1
gru_3_while_placeholder_2*
&gru_3_while_less_gru_3_strided_slice_1@
<gru_3_while_gru_3_while_cond_139569___redundant_placeholder0@
<gru_3_while_gru_3_while_cond_139569___redundant_placeholder1@
<gru_3_while_gru_3_while_cond_139569___redundant_placeholder2@
<gru_3_while_gru_3_while_cond_139569___redundant_placeholder3
gru_3_while_identity
z
gru_3/while/LessLessgru_3_while_placeholder&gru_3_while_less_gru_3_strided_slice_1*
T0*
_output_shapes
: W
gru_3/while/IdentityIdentitygru_3/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_3_while_identitygru_3/while/Identity:output:0*(
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
 
µ
while_body_138047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_11_138069_0:	¬-
while_gru_cell_11_138071_0:	d¬-
while_gru_cell_11_138073_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_11_138069:	¬+
while_gru_cell_11_138071:	d¬+
while_gru_cell_11_138073:	d¬¢)while/gru_cell_11/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
)while/gru_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_11_138069_0while_gru_cell_11_138071_0while_gru_cell_11_138073_0*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137995Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_11/StatefulPartitionedCall:output:0*
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
: :éèÒ
while/Identity_4Identity2while/gru_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx

while/NoOpNoOp*^while/gru_cell_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_11_138069while_gru_cell_11_138069_0"6
while_gru_cell_11_138071while_gru_cell_11_138071_0"6
while_gru_cell_11_138073while_gru_cell_11_138073_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2V
)while/gru_cell_11/StatefulPartitionedCall)while/gru_cell_11/StatefulPartitionedCall: 
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


#__inference_internal_grad_fn_143953
result_grads_0
result_grads_1
mul_while_gru_cell_10_beta
mul_while_gru_cell_10_add_2
identity
mulMulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2*
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


#__inference_internal_grad_fn_143917
result_grads_0
result_grads_1
mul_while_gru_cell_10_beta
mul_while_gru_cell_10_add_2
identity
mulMulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2*
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
÷B

while_body_138974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_10_readvariableop_resource_0:	¬E
2while_gru_cell_10_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_10_readvariableop_resource:	¬C
0while_gru_cell_10_matmul_readvariableop_resource:	d¬E
2while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_10/MatMul/ReadVariableOp¢)while/gru_cell_10/MatMul_1/ReadVariableOp¢ while/gru_cell_10/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_10/ReadVariableOpReadVariableOp+while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_10/unstackUnpack(while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_10/BiasAddBiasAdd"while/gru_cell_10/MatMul:product:0"while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_10/splitSplit*while/gru_cell_10/split/split_dim:output:0"while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_10/MatMul_1MatMulwhile_placeholder_21while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_10/BiasAdd_1BiasAdd$while/gru_cell_10/MatMul_1:product:0"while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_10/split_1SplitV$while/gru_cell_10/BiasAdd_1:output:0 while/gru_cell_10/Const:output:0,while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_10/addAddV2 while/gru_cell_10/split:output:0"while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_10/SigmoidSigmoidwhile/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_1AddV2 while/gru_cell_10/split:output:1"while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_1Sigmoidwhile/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mulMulwhile/gru_cell_10/Sigmoid_1:y:0"while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_2AddV2 while/gru_cell_10/split:output:2while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/mul_1Mulwhile/gru_cell_10/beta:output:0while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_2Sigmoidwhile/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_2Mulwhile/gru_cell_10/add_2:z:0while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/IdentityIdentitywhile/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_10/IdentityN	IdentityNwhile/gru_cell_10/mul_2:z:0while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139024*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_3Mulwhile/gru_cell_10/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/subSub while/gru_cell_10/sub/x:output:0while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_4Mulwhile/gru_cell_10/sub:z:0$while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_3AddV2while/gru_cell_10/mul_3:z:0while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_10/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_10/MatMul/ReadVariableOp*^while/gru_cell_10/MatMul_1/ReadVariableOp!^while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_10_matmul_1_readvariableop_resource4while_gru_cell_10_matmul_1_readvariableop_resource_0"f
0while_gru_cell_10_matmul_readvariableop_resource2while_gru_cell_10_matmul_readvariableop_resource_0"X
)while_gru_cell_10_readvariableop_resource+while_gru_cell_10_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_10/MatMul/ReadVariableOp'while/gru_cell_10/MatMul/ReadVariableOp2V
)while/gru_cell_10/MatMul_1/ReadVariableOp)while/gru_cell_10/MatMul_1/ReadVariableOp2D
 while/gru_cell_10/ReadVariableOp while/gru_cell_10/ReadVariableOp: 
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

x
#__inference_internal_grad_fn_144223
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
©
¨
#__inference_internal_grad_fn_143539
result_grads_0
result_grads_1$
 mul_gru_4_while_gru_cell_10_beta%
!mul_gru_4_while_gru_cell_10_add_2
identity
mulMul mul_gru_4_while_gru_cell_10_beta!mul_gru_4_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul mul_gru_4_while_gru_cell_10_beta!mul_gru_4_while_gru_cell_10_add_2*
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
K
¼	
gru_4_while_body_139733(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2'
#gru_4_while_gru_4_strided_slice_1_0c
_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0D
1gru_4_while_gru_cell_10_readvariableop_resource_0:	¬K
8gru_4_while_gru_cell_10_matmul_readvariableop_resource_0:	d¬M
:gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
gru_4_while_identity
gru_4_while_identity_1
gru_4_while_identity_2
gru_4_while_identity_3
gru_4_while_identity_4%
!gru_4_while_gru_4_strided_slice_1a
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensorB
/gru_4_while_gru_cell_10_readvariableop_resource:	¬I
6gru_4_while_gru_cell_10_matmul_readvariableop_resource:	d¬K
8gru_4_while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢-gru_4/while/gru_cell_10/MatMul/ReadVariableOp¢/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp¢&gru_4/while/gru_cell_10/ReadVariableOp
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0gru_4_while_placeholderFgru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
&gru_4/while/gru_cell_10/ReadVariableOpReadVariableOp1gru_4_while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_4/while/gru_cell_10/unstackUnpack.gru_4/while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num§
-gru_4/while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp8gru_4_while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ê
gru_4/while/gru_cell_10/MatMulMatMul6gru_4/while/TensorArrayV2Read/TensorListGetItem:item:05gru_4/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
gru_4/while/gru_cell_10/BiasAddBiasAdd(gru_4/while/gru_cell_10/MatMul:product:0(gru_4/while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
'gru_4/while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_4/while/gru_cell_10/splitSplit0gru_4/while/gru_cell_10/split/split_dim:output:0(gru_4/while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split«
/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp:gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0±
 gru_4/while/gru_cell_10/MatMul_1MatMulgru_4_while_placeholder_27gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬µ
!gru_4/while/gru_cell_10/BiasAdd_1BiasAdd*gru_4/while/gru_cell_10/MatMul_1:product:0(gru_4/while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
gru_4/while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿt
)gru_4/while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_4/while/gru_cell_10/split_1SplitV*gru_4/while/gru_cell_10/BiasAdd_1:output:0&gru_4/while/gru_cell_10/Const:output:02gru_4/while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¨
gru_4/while/gru_cell_10/addAddV2&gru_4/while/gru_cell_10/split:output:0(gru_4/while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_4/while/gru_cell_10/SigmoidSigmoidgru_4/while/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
gru_4/while/gru_cell_10/add_1AddV2&gru_4/while/gru_cell_10/split:output:1(gru_4/while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_4/while/gru_cell_10/Sigmoid_1Sigmoid!gru_4/while/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
gru_4/while/gru_cell_10/mulMul%gru_4/while/gru_cell_10/Sigmoid_1:y:0(gru_4/while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¡
gru_4/while/gru_cell_10/add_2AddV2&gru_4/while/gru_cell_10/split:output:2gru_4/while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_4/while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ? 
gru_4/while/gru_cell_10/mul_1Mul%gru_4/while/gru_cell_10/beta:output:0!gru_4/while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_4/while/gru_cell_10/Sigmoid_2Sigmoid!gru_4/while/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_4/while/gru_cell_10/mul_2Mul!gru_4/while/gru_cell_10/add_2:z:0%gru_4/while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_4/while/gru_cell_10/IdentityIdentity!gru_4/while/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdë
!gru_4/while/gru_cell_10/IdentityN	IdentityN!gru_4/while/gru_cell_10/mul_2:z:0!gru_4/while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139783*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_4/while/gru_cell_10/mul_3Mul#gru_4/while/gru_cell_10/Sigmoid:y:0gru_4_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
gru_4/while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_4/while/gru_cell_10/subSub&gru_4/while/gru_cell_10/sub/x:output:0#gru_4/while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd£
gru_4/while/gru_cell_10/mul_4Mulgru_4/while/gru_cell_10/sub:z:0*gru_4/while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/while/gru_cell_10/add_3AddV2!gru_4/while/gru_cell_10/mul_3:z:0!gru_4/while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÜ
0gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_4_while_placeholder_1gru_4_while_placeholder!gru_4/while/gru_cell_10/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_4/while/addAddV2gru_4_while_placeholdergru_4/while/add/y:output:0*
T0*
_output_shapes
: U
gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_4/while/add_1AddV2$gru_4_while_gru_4_while_loop_countergru_4/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_4/while/IdentityIdentitygru_4/while/add_1:z:0^gru_4/while/NoOp*
T0*
_output_shapes
: 
gru_4/while/Identity_1Identity*gru_4_while_gru_4_while_maximum_iterations^gru_4/while/NoOp*
T0*
_output_shapes
: k
gru_4/while/Identity_2Identitygru_4/while/add:z:0^gru_4/while/NoOp*
T0*
_output_shapes
: «
gru_4/while/Identity_3Identity@gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_4/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_4/while/Identity_4Identity!gru_4/while/gru_cell_10/add_3:z:0^gru_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÝ
gru_4/while/NoOpNoOp.^gru_4/while/gru_cell_10/MatMul/ReadVariableOp0^gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp'^gru_4/while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_4_while_gru_4_strided_slice_1#gru_4_while_gru_4_strided_slice_1_0"v
8gru_4_while_gru_cell_10_matmul_1_readvariableop_resource:gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0"r
6gru_4_while_gru_cell_10_matmul_readvariableop_resource8gru_4_while_gru_cell_10_matmul_readvariableop_resource_0"d
/gru_4_while_gru_cell_10_readvariableop_resource1gru_4_while_gru_cell_10_readvariableop_resource_0"5
gru_4_while_identitygru_4/while/Identity:output:0"9
gru_4_while_identity_1gru_4/while/Identity_1:output:0"9
gru_4_while_identity_2gru_4/while/Identity_2:output:0"9
gru_4_while_identity_3gru_4/while/Identity_3:output:0"9
gru_4_while_identity_4gru_4/while/Identity_4:output:0"À
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2^
-gru_4/while/gru_cell_10/MatMul/ReadVariableOp-gru_4/while/gru_cell_10/MatMul/ReadVariableOp2b
/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp2P
&gru_4/while/gru_cell_10/ReadVariableOp&gru_4/while/gru_cell_10/ReadVariableOp: 
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
B
ÿ
while_body_141142
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_9_readvariableop_resource_0:	¬D
1while_gru_cell_9_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_9_readvariableop_resource:	¬B
/while_gru_cell_9_matmul_readvariableop_resource:	¬D
1while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_9/MatMul/ReadVariableOp¢(while/gru_cell_9/MatMul_1/ReadVariableOp¢while/gru_cell_9/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0while/gru_cell_9/Const:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/beta:output:0while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_2Sigmoidwhile/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/add_2:z:0while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/IdentityIdentitywhile/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
while/gru_cell_9/IdentityN	IdentityNwhile/gru_cell_9/mul_2:z:0while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141192*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_3Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_4Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_3:z:0while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
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
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 
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
ÊQ

A__inference_gru_3_layer_call_and_return_conditional_losses_141238

inputs5
"gru_cell_9_readvariableop_resource:	¬<
)gru_cell_9_matmul_readvariableop_resource:	¬>
+gru_cell_9_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_9/MatMul/ReadVariableOp¢"gru_cell_9/MatMul_1/ReadVariableOp¢gru_cell_9/ReadVariableOp¢while;
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
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_9/mul_1Mulgru_cell_9/beta:output:0gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_2Sigmoidgru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_9/mul_2Mulgru_cell_9/add_2:z:0gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/IdentityIdentitygru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
gru_cell_9/IdentityN	IdentityNgru_cell_9/mul_2:z:0gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141126*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_9/mul_3Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_9/mul_4Mulgru_cell_9/sub:z:0gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_9/add_3AddV2gru_cell_9/mul_3:z:0gru_cell_9/mul_4:z:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
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
bodyR
while_body_141142*
condR
while_cond_141141*8
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
NoOpNoOp!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ö

­
-__inference_sequential_6_layer_call_fn_139379
gru_3_input
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
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallgru_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_139327o
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
_user_specified_namegru_3_input
ß

#__inference_internal_grad_fn_144079
result_grads_0
result_grads_1
mul_gru_cell_11_beta
mul_gru_cell_11_add_2
identityz
mulMulmul_gru_cell_11_betamul_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_11_betamul_gru_cell_11_add_2*
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

¹
&__inference_gru_5_layer_call_fn_141961
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallå
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_137922o
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
È\
Â
$sequential_6_gru_4_while_body_136799B
>sequential_6_gru_4_while_sequential_6_gru_4_while_loop_counterH
Dsequential_6_gru_4_while_sequential_6_gru_4_while_maximum_iterations(
$sequential_6_gru_4_while_placeholder*
&sequential_6_gru_4_while_placeholder_1*
&sequential_6_gru_4_while_placeholder_2A
=sequential_6_gru_4_while_sequential_6_gru_4_strided_slice_1_0}
ysequential_6_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_4_tensorarrayunstack_tensorlistfromtensor_0Q
>sequential_6_gru_4_while_gru_cell_10_readvariableop_resource_0:	¬X
Esequential_6_gru_4_while_gru_cell_10_matmul_readvariableop_resource_0:	d¬Z
Gsequential_6_gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬%
!sequential_6_gru_4_while_identity'
#sequential_6_gru_4_while_identity_1'
#sequential_6_gru_4_while_identity_2'
#sequential_6_gru_4_while_identity_3'
#sequential_6_gru_4_while_identity_4?
;sequential_6_gru_4_while_sequential_6_gru_4_strided_slice_1{
wsequential_6_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_4_tensorarrayunstack_tensorlistfromtensorO
<sequential_6_gru_4_while_gru_cell_10_readvariableop_resource:	¬V
Csequential_6_gru_4_while_gru_cell_10_matmul_readvariableop_resource:	d¬X
Esequential_6_gru_4_while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢:sequential_6/gru_4/while/gru_cell_10/MatMul/ReadVariableOp¢<sequential_6/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp¢3sequential_6/gru_4/while/gru_cell_10/ReadVariableOp
Jsequential_6/gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
<sequential_6/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_6_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_4_tensorarrayunstack_tensorlistfromtensor_0$sequential_6_gru_4_while_placeholderSsequential_6/gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0³
3sequential_6/gru_4/while/gru_cell_10/ReadVariableOpReadVariableOp>sequential_6_gru_4_while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0«
,sequential_6/gru_4/while/gru_cell_10/unstackUnpack;sequential_6/gru_4/while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numÁ
:sequential_6/gru_4/while/gru_cell_10/MatMul/ReadVariableOpReadVariableOpEsequential_6_gru_4_while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0ñ
+sequential_6/gru_4/while/gru_cell_10/MatMulMatMulCsequential_6/gru_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_6/gru_4/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ø
,sequential_6/gru_4/while/gru_cell_10/BiasAddBiasAdd5sequential_6/gru_4/while/gru_cell_10/MatMul:product:05sequential_6/gru_4/while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
4sequential_6/gru_4/while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
*sequential_6/gru_4/while/gru_cell_10/splitSplit=sequential_6/gru_4/while/gru_cell_10/split/split_dim:output:05sequential_6/gru_4/while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÅ
<sequential_6/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOpGsequential_6_gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ø
-sequential_6/gru_4/while/gru_cell_10/MatMul_1MatMul&sequential_6_gru_4_while_placeholder_2Dsequential_6/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ü
.sequential_6/gru_4/while/gru_cell_10/BiasAdd_1BiasAdd7sequential_6/gru_4/while/gru_cell_10/MatMul_1:product:05sequential_6/gru_4/while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
*sequential_6/gru_4/while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ
6sequential_6/gru_4/while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
,sequential_6/gru_4/while/gru_cell_10/split_1SplitV7sequential_6/gru_4/while/gru_cell_10/BiasAdd_1:output:03sequential_6/gru_4/while/gru_cell_10/Const:output:0?sequential_6/gru_4/while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÏ
(sequential_6/gru_4/while/gru_cell_10/addAddV23sequential_6/gru_4/while/gru_cell_10/split:output:05sequential_6/gru_4/while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_6/gru_4/while/gru_cell_10/SigmoidSigmoid,sequential_6/gru_4/while/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÑ
*sequential_6/gru_4/while/gru_cell_10/add_1AddV23sequential_6/gru_4/while/gru_cell_10/split:output:15sequential_6/gru_4/while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
.sequential_6/gru_4/while/gru_cell_10/Sigmoid_1Sigmoid.sequential_6/gru_4/while/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÌ
(sequential_6/gru_4/while/gru_cell_10/mulMul2sequential_6/gru_4/while/gru_cell_10/Sigmoid_1:y:05sequential_6/gru_4/while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÈ
*sequential_6/gru_4/while/gru_cell_10/add_2AddV23sequential_6/gru_4/while/gru_cell_10/split:output:2,sequential_6/gru_4/while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
)sequential_6/gru_4/while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
*sequential_6/gru_4/while/gru_cell_10/mul_1Mul2sequential_6/gru_4/while/gru_cell_10/beta:output:0.sequential_6/gru_4/while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
.sequential_6/gru_4/while/gru_cell_10/Sigmoid_2Sigmoid.sequential_6/gru_4/while/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
*sequential_6/gru_4/while/gru_cell_10/mul_2Mul.sequential_6/gru_4/while/gru_cell_10/add_2:z:02sequential_6/gru_4/while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_6/gru_4/while/gru_cell_10/IdentityIdentity.sequential_6/gru_4/while/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
.sequential_6/gru_4/while/gru_cell_10/IdentityN	IdentityN.sequential_6/gru_4/while/gru_cell_10/mul_2:z:0.sequential_6/gru_4/while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-136849*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd½
*sequential_6/gru_4/while/gru_cell_10/mul_3Mul0sequential_6/gru_4/while/gru_cell_10/Sigmoid:y:0&sequential_6_gru_4_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
*sequential_6/gru_4/while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
(sequential_6/gru_4/while/gru_cell_10/subSub3sequential_6/gru_4/while/gru_cell_10/sub/x:output:00sequential_6/gru_4/while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÊ
*sequential_6/gru_4/while/gru_cell_10/mul_4Mul,sequential_6/gru_4/while/gru_cell_10/sub:z:07sequential_6/gru_4/while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ
*sequential_6/gru_4/while/gru_cell_10/add_3AddV2.sequential_6/gru_4/while/gru_cell_10/mul_3:z:0.sequential_6/gru_4/while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
=sequential_6/gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_6_gru_4_while_placeholder_1$sequential_6_gru_4_while_placeholder.sequential_6/gru_4/while/gru_cell_10/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_6/gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_6/gru_4/while/addAddV2$sequential_6_gru_4_while_placeholder'sequential_6/gru_4/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_6/gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_6/gru_4/while/add_1AddV2>sequential_6_gru_4_while_sequential_6_gru_4_while_loop_counter)sequential_6/gru_4/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_6/gru_4/while/IdentityIdentity"sequential_6/gru_4/while/add_1:z:0^sequential_6/gru_4/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_6/gru_4/while/Identity_1IdentityDsequential_6_gru_4_while_sequential_6_gru_4_while_maximum_iterations^sequential_6/gru_4/while/NoOp*
T0*
_output_shapes
: 
#sequential_6/gru_4/while/Identity_2Identity sequential_6/gru_4/while/add:z:0^sequential_6/gru_4/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_6/gru_4/while/Identity_3IdentityMsequential_6/gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_6/gru_4/while/NoOp*
T0*
_output_shapes
: :éèÒ±
#sequential_6/gru_4/while/Identity_4Identity.sequential_6/gru_4/while/gru_cell_10/add_3:z:0^sequential_6/gru_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_6/gru_4/while/NoOpNoOp;^sequential_6/gru_4/while/gru_cell_10/MatMul/ReadVariableOp=^sequential_6/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp4^sequential_6/gru_4/while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Esequential_6_gru_4_while_gru_cell_10_matmul_1_readvariableop_resourceGsequential_6_gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0"
Csequential_6_gru_4_while_gru_cell_10_matmul_readvariableop_resourceEsequential_6_gru_4_while_gru_cell_10_matmul_readvariableop_resource_0"~
<sequential_6_gru_4_while_gru_cell_10_readvariableop_resource>sequential_6_gru_4_while_gru_cell_10_readvariableop_resource_0"O
!sequential_6_gru_4_while_identity*sequential_6/gru_4/while/Identity:output:0"S
#sequential_6_gru_4_while_identity_1,sequential_6/gru_4/while/Identity_1:output:0"S
#sequential_6_gru_4_while_identity_2,sequential_6/gru_4/while/Identity_2:output:0"S
#sequential_6_gru_4_while_identity_3,sequential_6/gru_4/while/Identity_3:output:0"S
#sequential_6_gru_4_while_identity_4,sequential_6/gru_4/while/Identity_4:output:0"|
;sequential_6_gru_4_while_sequential_6_gru_4_strided_slice_1=sequential_6_gru_4_while_sequential_6_gru_4_strided_slice_1_0"ô
wsequential_6_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_4_tensorarrayunstack_tensorlistfromtensorysequential_6_gru_4_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2x
:sequential_6/gru_4/while/gru_cell_10/MatMul/ReadVariableOp:sequential_6/gru_4/while/gru_cell_10/MatMul/ReadVariableOp2|
<sequential_6/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp<sequential_6/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp2j
3sequential_6/gru_4/while/gru_cell_10/ReadVariableOp3sequential_6/gru_4/while/gru_cell_10/ReadVariableOp: 
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
 
µ
while_body_137506
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_10_137528_0:	¬-
while_gru_cell_10_137530_0:	d¬-
while_gru_cell_10_137532_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_10_137528:	¬+
while_gru_cell_10_137530:	d¬+
while_gru_cell_10_137532:	d¬¢)while/gru_cell_10/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
)while/gru_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_10_137528_0while_gru_cell_10_137530_0while_gru_cell_10_137532_0*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137493Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_10/StatefulPartitionedCall:output:0*
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
: :éèÒ
while/Identity_4Identity2while/gru_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx

while/NoOpNoOp*^while/gru_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_10_137528while_gru_cell_10_137528_0"6
while_gru_cell_10_137530while_gru_cell_10_137530_0"6
while_gru_cell_10_137532while_gru_cell_10_137532_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2V
)while/gru_cell_10/StatefulPartitionedCall)while/gru_cell_10/StatefulPartitionedCall: 
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
©
¹
&__inference_gru_3_layer_call_fn_140537
inputs_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallò
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_137218|
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
Ú
ª
while_cond_138973
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_138973___redundant_placeholder04
0while_while_cond_138973___redundant_placeholder14
0while_while_cond_138973___redundant_placeholder24
0while_while_cond_138973___redundant_placeholder3
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
 
µ
while_body_137858
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_11_137880_0:	¬-
while_gru_cell_11_137882_0:	d¬-
while_gru_cell_11_137884_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_11_137880:	¬+
while_gru_cell_11_137882:	d¬+
while_gru_cell_11_137884:	d¬¢)while/gru_cell_11/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
)while/gru_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_11_137880_0while_gru_cell_11_137882_0while_gru_cell_11_137884_0*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137845Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_11/StatefulPartitionedCall:output:0*
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
: :éèÒ
while/Identity_4Identity2while/gru_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx

while/NoOpNoOp*^while/gru_cell_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_11_137880while_gru_cell_11_137880_0"6
while_gru_cell_11_137882while_gru_cell_11_137882_0"6
while_gru_cell_11_137884while_gru_cell_11_137884_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2V
)while/gru_cell_11/StatefulPartitionedCall)while/gru_cell_11/StatefulPartitionedCall: 
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
Ú
ª
while_cond_142398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_142398___redundant_placeholder04
0while_while_cond_142398___redundant_placeholder14
0while_while_cond_142398___redundant_placeholder24
0while_while_cond_142398___redundant_placeholder3
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
´

Û
,__inference_gru_cell_10_layer_call_fn_142815

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137493o
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
ß

#__inference_internal_grad_fn_144151
result_grads_0
result_grads_1
mul_gru_cell_11_beta
mul_gru_cell_11_add_2
identityz
mulMulmul_gru_cell_11_betamul_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_11_betamul_gru_cell_11_add_2*
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
þ

#__inference_internal_grad_fn_143467
result_grads_0
result_grads_1
mul_gru_3_gru_cell_9_beta
mul_gru_3_gru_cell_9_add_2
identity
mulMulmul_gru_3_gru_cell_9_betamul_gru_3_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_3_gru_cell_9_betamul_gru_3_gru_cell_9_add_2*
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

¹
&__inference_gru_5_layer_call_fn_141972
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallå
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138111o
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
½


H__inference_sequential_6_layer_call_and_return_conditional_losses_140497

inputs;
(gru_3_gru_cell_9_readvariableop_resource:	¬B
/gru_3_gru_cell_9_matmul_readvariableop_resource:	¬D
1gru_3_gru_cell_9_matmul_1_readvariableop_resource:	d¬<
)gru_4_gru_cell_10_readvariableop_resource:	¬C
0gru_4_gru_cell_10_matmul_readvariableop_resource:	d¬E
2gru_4_gru_cell_10_matmul_1_readvariableop_resource:	d¬<
)gru_5_gru_cell_11_readvariableop_resource:	¬C
0gru_5_gru_cell_11_matmul_readvariableop_resource:	d¬E
2gru_5_gru_cell_11_matmul_1_readvariableop_resource:	d¬8
&dense_6_matmul_readvariableop_resource:d5
'dense_6_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢&gru_3/gru_cell_9/MatMul/ReadVariableOp¢(gru_3/gru_cell_9/MatMul_1/ReadVariableOp¢gru_3/gru_cell_9/ReadVariableOp¢gru_3/while¢'gru_4/gru_cell_10/MatMul/ReadVariableOp¢)gru_4/gru_cell_10/MatMul_1/ReadVariableOp¢ gru_4/gru_cell_10/ReadVariableOp¢gru_4/while¢'gru_5/gru_cell_11/MatMul/ReadVariableOp¢)gru_5/gru_cell_11/MatMul_1/ReadVariableOp¢ gru_5/gru_cell_11/ReadVariableOp¢gru_5/whileA
gru_3/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_3/strided_sliceStridedSlicegru_3/Shape:output:0"gru_3/strided_slice/stack:output:0$gru_3/strided_slice/stack_1:output:0$gru_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_3/zeros/packedPackgru_3/strided_slice:output:0gru_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_3/zerosFillgru_3/zeros/packed:output:0gru_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_3/transpose	Transposeinputsgru_3/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿP
gru_3/Shape_1Shapegru_3/transpose:y:0*
T0*
_output_shapes
:e
gru_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_3/strided_slice_1StridedSlicegru_3/Shape_1:output:0$gru_3/strided_slice_1/stack:output:0&gru_3/strided_slice_1/stack_1:output:0&gru_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_3/TensorArrayV2TensorListReserve*gru_3/TensorArrayV2/element_shape:output:0gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_3/transpose:y:0Dgru_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_3/strided_slice_2StridedSlicegru_3/transpose:y:0$gru_3/strided_slice_2/stack:output:0&gru_3/strided_slice_2/stack_1:output:0&gru_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_3/gru_cell_9/ReadVariableOpReadVariableOp(gru_3_gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_3/gru_cell_9/unstackUnpack'gru_3/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_3/gru_cell_9/MatMul/ReadVariableOpReadVariableOp/gru_3_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0¤
gru_3/gru_cell_9/MatMulMatMulgru_3/strided_slice_2:output:0.gru_3/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_3/gru_cell_9/BiasAddBiasAdd!gru_3/gru_cell_9/MatMul:product:0!gru_3/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_3/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_3/gru_cell_9/splitSplit)gru_3/gru_cell_9/split/split_dim:output:0!gru_3/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_3/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp1gru_3_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_3/gru_cell_9/MatMul_1MatMulgru_3/zeros:output:00gru_3/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_3/gru_cell_9/BiasAdd_1BiasAdd#gru_3/gru_cell_9/MatMul_1:product:0!gru_3/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_3/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_3/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_3/gru_cell_9/split_1SplitV#gru_3/gru_cell_9/BiasAdd_1:output:0gru_3/gru_cell_9/Const:output:0+gru_3/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_3/gru_cell_9/addAddV2gru_3/gru_cell_9/split:output:0!gru_3/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_3/gru_cell_9/SigmoidSigmoidgru_3/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/add_1AddV2gru_3/gru_cell_9/split:output:1!gru_3/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_3/gru_cell_9/Sigmoid_1Sigmoidgru_3/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mulMulgru_3/gru_cell_9/Sigmoid_1:y:0!gru_3/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/add_2AddV2gru_3/gru_cell_9/split:output:2gru_3/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_3/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/gru_cell_9/mul_1Mulgru_3/gru_cell_9/beta:output:0gru_3/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_3/gru_cell_9/Sigmoid_2Sigmoidgru_3/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mul_2Mulgru_3/gru_cell_9/add_2:z:0gru_3/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_3/gru_cell_9/IdentityIdentitygru_3/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
gru_3/gru_cell_9/IdentityN	IdentityNgru_3/gru_cell_9/mul_2:z:0gru_3/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140053*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mul_3Mulgru_3/gru_cell_9/Sigmoid:y:0gru_3/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_3/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/gru_cell_9/subSubgru_3/gru_cell_9/sub/x:output:0gru_3/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mul_4Mulgru_3/gru_cell_9/sub:z:0#gru_3/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/add_3AddV2gru_3/gru_cell_9/mul_3:z:0gru_3/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_3/TensorArrayV2_1TensorListReserve,gru_3/TensorArrayV2_1/element_shape:output:0gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_3/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_3/whileWhile!gru_3/while/loop_counter:output:0'gru_3/while/maximum_iterations:output:0gru_3/time:output:0gru_3/TensorArrayV2_1:handle:0gru_3/zeros:output:0gru_3/strided_slice_1:output:0=gru_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_3_gru_cell_9_readvariableop_resource/gru_3_gru_cell_9_matmul_readvariableop_resource1gru_3_gru_cell_9_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *#
bodyR
gru_3_while_body_140069*#
condR
gru_3_while_cond_140068*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_3/TensorArrayV2Stack/TensorListStackTensorListStackgru_3/while:output:3?gru_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_3/strided_slice_3StridedSlice1gru_3/TensorArrayV2Stack/TensorListStack:tensor:0$gru_3/strided_slice_3/stack:output:0&gru_3/strided_slice_3/stack_1:output:0&gru_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_3/transpose_1	Transpose1gru_3/TensorArrayV2Stack/TensorListStack:tensor:0gru_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_4/ShapeShapegru_3/transpose_1:y:0*
T0*
_output_shapes
:c
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_4/transpose	Transposegru_3/transpose_1:y:0gru_4/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_4/Shape_1Shapegru_4/transpose:y:0*
T0*
_output_shapes
:e
gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_4/strided_slice_1StridedSlicegru_4/Shape_1:output:0$gru_4/strided_slice_1/stack:output:0&gru_4/strided_slice_1/stack_1:output:0&gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_4/TensorArrayV2TensorListReserve*gru_4/TensorArrayV2/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_4/transpose:y:0Dgru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_4/strided_slice_2StridedSlicegru_4/transpose:y:0$gru_4/strided_slice_2/stack:output:0&gru_4/strided_slice_2/stack_1:output:0&gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
 gru_4/gru_cell_10/ReadVariableOpReadVariableOp)gru_4_gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_4/gru_cell_10/unstackUnpack(gru_4/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'gru_4/gru_cell_10/MatMul/ReadVariableOpReadVariableOp0gru_4_gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¦
gru_4/gru_cell_10/MatMulMatMulgru_4/strided_slice_2:output:0/gru_4/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_4/gru_cell_10/BiasAddBiasAdd"gru_4/gru_cell_10/MatMul:product:0"gru_4/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!gru_4/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_4/gru_cell_10/splitSplit*gru_4/gru_cell_10/split/split_dim:output:0"gru_4/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)gru_4/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp2gru_4_gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0 
gru_4/gru_cell_10/MatMul_1MatMulgru_4/zeros:output:01gru_4/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
gru_4/gru_cell_10/BiasAdd_1BiasAdd$gru_4/gru_cell_10/MatMul_1:product:0"gru_4/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
gru_4/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#gru_4/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_4/gru_cell_10/split_1SplitV$gru_4/gru_cell_10/BiasAdd_1:output:0 gru_4/gru_cell_10/Const:output:0,gru_4/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_4/gru_cell_10/addAddV2 gru_4/gru_cell_10/split:output:0"gru_4/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_4/gru_cell_10/SigmoidSigmoidgru_4/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/add_1AddV2 gru_4/gru_cell_10/split:output:1"gru_4/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_4/gru_cell_10/Sigmoid_1Sigmoidgru_4/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mulMulgru_4/gru_cell_10/Sigmoid_1:y:0"gru_4/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/add_2AddV2 gru_4/gru_cell_10/split:output:2gru_4/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_4/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_4/gru_cell_10/mul_1Mulgru_4/gru_cell_10/beta:output:0gru_4/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_4/gru_cell_10/Sigmoid_2Sigmoidgru_4/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mul_2Mulgru_4/gru_cell_10/add_2:z:0gru_4/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_4/gru_cell_10/IdentityIdentitygru_4/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
gru_4/gru_cell_10/IdentityN	IdentityNgru_4/gru_cell_10/mul_2:z:0gru_4/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140216*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mul_3Mulgru_4/gru_cell_10/Sigmoid:y:0gru_4/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
gru_4/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_4/gru_cell_10/subSub gru_4/gru_cell_10/sub/x:output:0gru_4/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mul_4Mulgru_4/gru_cell_10/sub:z:0$gru_4/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/add_3AddV2gru_4/gru_cell_10/mul_3:z:0gru_4/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_4/TensorArrayV2_1TensorListReserve,gru_4/TensorArrayV2_1/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_4/whileWhile!gru_4/while/loop_counter:output:0'gru_4/while/maximum_iterations:output:0gru_4/time:output:0gru_4/TensorArrayV2_1:handle:0gru_4/zeros:output:0gru_4/strided_slice_1:output:0=gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_4_gru_cell_10_readvariableop_resource0gru_4_gru_cell_10_matmul_readvariableop_resource2gru_4_gru_cell_10_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *#
bodyR
gru_4_while_body_140232*#
condR
gru_4_while_cond_140231*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_4/TensorArrayV2Stack/TensorListStackTensorListStackgru_4/while:output:3?gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_4/strided_slice_3StridedSlice1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0$gru_4/strided_slice_3/stack:output:0&gru_4/strided_slice_3/stack_1:output:0&gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_4/transpose_1	Transpose1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_5/ShapeShapegru_4/transpose_1:y:0*
T0*
_output_shapes
:c
gru_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_5/strided_sliceStridedSlicegru_5/Shape:output:0"gru_5/strided_slice/stack:output:0$gru_5/strided_slice/stack_1:output:0$gru_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_5/zeros/packedPackgru_5/strided_slice:output:0gru_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_5/zerosFillgru_5/zeros/packed:output:0gru_5/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_5/transpose	Transposegru_4/transpose_1:y:0gru_5/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_5/Shape_1Shapegru_5/transpose:y:0*
T0*
_output_shapes
:e
gru_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_5/strided_slice_1StridedSlicegru_5/Shape_1:output:0$gru_5/strided_slice_1/stack:output:0&gru_5/strided_slice_1/stack_1:output:0&gru_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_5/TensorArrayV2TensorListReserve*gru_5/TensorArrayV2/element_shape:output:0gru_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_5/transpose:y:0Dgru_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_5/strided_slice_2StridedSlicegru_5/transpose:y:0$gru_5/strided_slice_2/stack:output:0&gru_5/strided_slice_2/stack_1:output:0&gru_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
 gru_5/gru_cell_11/ReadVariableOpReadVariableOp)gru_5_gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_5/gru_cell_11/unstackUnpack(gru_5/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'gru_5/gru_cell_11/MatMul/ReadVariableOpReadVariableOp0gru_5_gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¦
gru_5/gru_cell_11/MatMulMatMulgru_5/strided_slice_2:output:0/gru_5/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_5/gru_cell_11/BiasAddBiasAdd"gru_5/gru_cell_11/MatMul:product:0"gru_5/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!gru_5/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_5/gru_cell_11/splitSplit*gru_5/gru_cell_11/split/split_dim:output:0"gru_5/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)gru_5/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp2gru_5_gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0 
gru_5/gru_cell_11/MatMul_1MatMulgru_5/zeros:output:01gru_5/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
gru_5/gru_cell_11/BiasAdd_1BiasAdd$gru_5/gru_cell_11/MatMul_1:product:0"gru_5/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
gru_5/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#gru_5/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_5/gru_cell_11/split_1SplitV$gru_5/gru_cell_11/BiasAdd_1:output:0 gru_5/gru_cell_11/Const:output:0,gru_5/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_5/gru_cell_11/addAddV2 gru_5/gru_cell_11/split:output:0"gru_5/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_5/gru_cell_11/SigmoidSigmoidgru_5/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/add_1AddV2 gru_5/gru_cell_11/split:output:1"gru_5/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_5/gru_cell_11/Sigmoid_1Sigmoidgru_5/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mulMulgru_5/gru_cell_11/Sigmoid_1:y:0"gru_5/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/add_2AddV2 gru_5/gru_cell_11/split:output:2gru_5/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_5/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_5/gru_cell_11/mul_1Mulgru_5/gru_cell_11/beta:output:0gru_5/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_5/gru_cell_11/Sigmoid_2Sigmoidgru_5/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mul_2Mulgru_5/gru_cell_11/add_2:z:0gru_5/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_5/gru_cell_11/IdentityIdentitygru_5/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
gru_5/gru_cell_11/IdentityN	IdentityNgru_5/gru_cell_11/mul_2:z:0gru_5/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140379*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mul_3Mulgru_5/gru_cell_11/Sigmoid:y:0gru_5/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
gru_5/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_5/gru_cell_11/subSub gru_5/gru_cell_11/sub/x:output:0gru_5/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mul_4Mulgru_5/gru_cell_11/sub:z:0$gru_5/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/add_3AddV2gru_5/gru_cell_11/mul_3:z:0gru_5/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_5/TensorArrayV2_1TensorListReserve,gru_5/TensorArrayV2_1/element_shape:output:0gru_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_5/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_5/whileWhile!gru_5/while/loop_counter:output:0'gru_5/while/maximum_iterations:output:0gru_5/time:output:0gru_5/TensorArrayV2_1:handle:0gru_5/zeros:output:0gru_5/strided_slice_1:output:0=gru_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_5_gru_cell_11_readvariableop_resource0gru_5_gru_cell_11_matmul_readvariableop_resource2gru_5_gru_cell_11_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *#
bodyR
gru_5_while_body_140395*#
condR
gru_5_while_cond_140394*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_5/TensorArrayV2Stack/TensorListStackTensorListStackgru_5/while:output:3?gru_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_5/strided_slice_3StridedSlice1gru_5/TensorArrayV2Stack/TensorListStack:tensor:0$gru_5/strided_slice_3/stack:output:0&gru_5/strided_slice_3/stack_1:output:0&gru_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_5/transpose_1	Transpose1gru_5/TensorArrayV2Stack/TensorListStack:tensor:0gru_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_6/MatMulMatMulgru_5/strided_slice_3:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp'^gru_3/gru_cell_9/MatMul/ReadVariableOp)^gru_3/gru_cell_9/MatMul_1/ReadVariableOp ^gru_3/gru_cell_9/ReadVariableOp^gru_3/while(^gru_4/gru_cell_10/MatMul/ReadVariableOp*^gru_4/gru_cell_10/MatMul_1/ReadVariableOp!^gru_4/gru_cell_10/ReadVariableOp^gru_4/while(^gru_5/gru_cell_11/MatMul/ReadVariableOp*^gru_5/gru_cell_11/MatMul_1/ReadVariableOp!^gru_5/gru_cell_11/ReadVariableOp^gru_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2P
&gru_3/gru_cell_9/MatMul/ReadVariableOp&gru_3/gru_cell_9/MatMul/ReadVariableOp2T
(gru_3/gru_cell_9/MatMul_1/ReadVariableOp(gru_3/gru_cell_9/MatMul_1/ReadVariableOp2B
gru_3/gru_cell_9/ReadVariableOpgru_3/gru_cell_9/ReadVariableOp2
gru_3/whilegru_3/while2R
'gru_4/gru_cell_10/MatMul/ReadVariableOp'gru_4/gru_cell_10/MatMul/ReadVariableOp2V
)gru_4/gru_cell_10/MatMul_1/ReadVariableOp)gru_4/gru_cell_10/MatMul_1/ReadVariableOp2D
 gru_4/gru_cell_10/ReadVariableOp gru_4/gru_cell_10/ReadVariableOp2
gru_4/whilegru_4/while2R
'gru_5/gru_cell_11/MatMul/ReadVariableOp'gru_5/gru_cell_11/MatMul/ReadVariableOp2V
)gru_5/gru_cell_11/MatMul_1/ReadVariableOp)gru_5/gru_cell_11/MatMul_1/ReadVariableOp2D
 gru_5/gru_cell_11/ReadVariableOp gru_5/gru_cell_11/ReadVariableOp2
gru_5/whilegru_5/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ü

gru_4_while_cond_139732(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2*
&gru_4_while_less_gru_4_strided_slice_1@
<gru_4_while_gru_4_while_cond_139732___redundant_placeholder0@
<gru_4_while_gru_4_while_cond_139732___redundant_placeholder1@
<gru_4_while_gru_4_while_cond_139732___redundant_placeholder2@
<gru_4_while_gru_4_while_cond_139732___redundant_placeholder3
gru_4_while_identity
z
gru_4/while/LessLessgru_4_while_placeholder&gru_4_while_less_gru_4_strided_slice_1*
T0*
_output_shapes
: W
gru_4/while/IdentityIdentitygru_4/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_4_while_identitygru_4/while/Identity:output:0*(
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
È\
Â
$sequential_6_gru_5_while_body_136962B
>sequential_6_gru_5_while_sequential_6_gru_5_while_loop_counterH
Dsequential_6_gru_5_while_sequential_6_gru_5_while_maximum_iterations(
$sequential_6_gru_5_while_placeholder*
&sequential_6_gru_5_while_placeholder_1*
&sequential_6_gru_5_while_placeholder_2A
=sequential_6_gru_5_while_sequential_6_gru_5_strided_slice_1_0}
ysequential_6_gru_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_5_tensorarrayunstack_tensorlistfromtensor_0Q
>sequential_6_gru_5_while_gru_cell_11_readvariableop_resource_0:	¬X
Esequential_6_gru_5_while_gru_cell_11_matmul_readvariableop_resource_0:	d¬Z
Gsequential_6_gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬%
!sequential_6_gru_5_while_identity'
#sequential_6_gru_5_while_identity_1'
#sequential_6_gru_5_while_identity_2'
#sequential_6_gru_5_while_identity_3'
#sequential_6_gru_5_while_identity_4?
;sequential_6_gru_5_while_sequential_6_gru_5_strided_slice_1{
wsequential_6_gru_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_5_tensorarrayunstack_tensorlistfromtensorO
<sequential_6_gru_5_while_gru_cell_11_readvariableop_resource:	¬V
Csequential_6_gru_5_while_gru_cell_11_matmul_readvariableop_resource:	d¬X
Esequential_6_gru_5_while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢:sequential_6/gru_5/while/gru_cell_11/MatMul/ReadVariableOp¢<sequential_6/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp¢3sequential_6/gru_5/while/gru_cell_11/ReadVariableOp
Jsequential_6/gru_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
<sequential_6/gru_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_6_gru_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_5_tensorarrayunstack_tensorlistfromtensor_0$sequential_6_gru_5_while_placeholderSsequential_6/gru_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0³
3sequential_6/gru_5/while/gru_cell_11/ReadVariableOpReadVariableOp>sequential_6_gru_5_while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0«
,sequential_6/gru_5/while/gru_cell_11/unstackUnpack;sequential_6/gru_5/while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
numÁ
:sequential_6/gru_5/while/gru_cell_11/MatMul/ReadVariableOpReadVariableOpEsequential_6_gru_5_while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0ñ
+sequential_6/gru_5/while/gru_cell_11/MatMulMatMulCsequential_6/gru_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_6/gru_5/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ø
,sequential_6/gru_5/while/gru_cell_11/BiasAddBiasAdd5sequential_6/gru_5/while/gru_cell_11/MatMul:product:05sequential_6/gru_5/while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
4sequential_6/gru_5/while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
*sequential_6/gru_5/while/gru_cell_11/splitSplit=sequential_6/gru_5/while/gru_cell_11/split/split_dim:output:05sequential_6/gru_5/while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÅ
<sequential_6/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOpGsequential_6_gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ø
-sequential_6/gru_5/while/gru_cell_11/MatMul_1MatMul&sequential_6_gru_5_while_placeholder_2Dsequential_6/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ü
.sequential_6/gru_5/while/gru_cell_11/BiasAdd_1BiasAdd7sequential_6/gru_5/while/gru_cell_11/MatMul_1:product:05sequential_6/gru_5/while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
*sequential_6/gru_5/while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ
6sequential_6/gru_5/while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
,sequential_6/gru_5/while/gru_cell_11/split_1SplitV7sequential_6/gru_5/while/gru_cell_11/BiasAdd_1:output:03sequential_6/gru_5/while/gru_cell_11/Const:output:0?sequential_6/gru_5/while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÏ
(sequential_6/gru_5/while/gru_cell_11/addAddV23sequential_6/gru_5/while/gru_cell_11/split:output:05sequential_6/gru_5/while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_6/gru_5/while/gru_cell_11/SigmoidSigmoid,sequential_6/gru_5/while/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÑ
*sequential_6/gru_5/while/gru_cell_11/add_1AddV23sequential_6/gru_5/while/gru_cell_11/split:output:15sequential_6/gru_5/while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
.sequential_6/gru_5/while/gru_cell_11/Sigmoid_1Sigmoid.sequential_6/gru_5/while/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÌ
(sequential_6/gru_5/while/gru_cell_11/mulMul2sequential_6/gru_5/while/gru_cell_11/Sigmoid_1:y:05sequential_6/gru_5/while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÈ
*sequential_6/gru_5/while/gru_cell_11/add_2AddV23sequential_6/gru_5/while/gru_cell_11/split:output:2,sequential_6/gru_5/while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
)sequential_6/gru_5/while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
*sequential_6/gru_5/while/gru_cell_11/mul_1Mul2sequential_6/gru_5/while/gru_cell_11/beta:output:0.sequential_6/gru_5/while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
.sequential_6/gru_5/while/gru_cell_11/Sigmoid_2Sigmoid.sequential_6/gru_5/while/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
*sequential_6/gru_5/while/gru_cell_11/mul_2Mul.sequential_6/gru_5/while/gru_cell_11/add_2:z:02sequential_6/gru_5/while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_6/gru_5/while/gru_cell_11/IdentityIdentity.sequential_6/gru_5/while/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
.sequential_6/gru_5/while/gru_cell_11/IdentityN	IdentityN.sequential_6/gru_5/while/gru_cell_11/mul_2:z:0.sequential_6/gru_5/while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137012*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd½
*sequential_6/gru_5/while/gru_cell_11/mul_3Mul0sequential_6/gru_5/while/gru_cell_11/Sigmoid:y:0&sequential_6_gru_5_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
*sequential_6/gru_5/while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
(sequential_6/gru_5/while/gru_cell_11/subSub3sequential_6/gru_5/while/gru_cell_11/sub/x:output:00sequential_6/gru_5/while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÊ
*sequential_6/gru_5/while/gru_cell_11/mul_4Mul,sequential_6/gru_5/while/gru_cell_11/sub:z:07sequential_6/gru_5/while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ
*sequential_6/gru_5/while/gru_cell_11/add_3AddV2.sequential_6/gru_5/while/gru_cell_11/mul_3:z:0.sequential_6/gru_5/while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
=sequential_6/gru_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_6_gru_5_while_placeholder_1$sequential_6_gru_5_while_placeholder.sequential_6/gru_5/while/gru_cell_11/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_6/gru_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_6/gru_5/while/addAddV2$sequential_6_gru_5_while_placeholder'sequential_6/gru_5/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_6/gru_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_6/gru_5/while/add_1AddV2>sequential_6_gru_5_while_sequential_6_gru_5_while_loop_counter)sequential_6/gru_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_6/gru_5/while/IdentityIdentity"sequential_6/gru_5/while/add_1:z:0^sequential_6/gru_5/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_6/gru_5/while/Identity_1IdentityDsequential_6_gru_5_while_sequential_6_gru_5_while_maximum_iterations^sequential_6/gru_5/while/NoOp*
T0*
_output_shapes
: 
#sequential_6/gru_5/while/Identity_2Identity sequential_6/gru_5/while/add:z:0^sequential_6/gru_5/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_6/gru_5/while/Identity_3IdentityMsequential_6/gru_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_6/gru_5/while/NoOp*
T0*
_output_shapes
: :éèÒ±
#sequential_6/gru_5/while/Identity_4Identity.sequential_6/gru_5/while/gru_cell_11/add_3:z:0^sequential_6/gru_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_6/gru_5/while/NoOpNoOp;^sequential_6/gru_5/while/gru_cell_11/MatMul/ReadVariableOp=^sequential_6/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp4^sequential_6/gru_5/while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Esequential_6_gru_5_while_gru_cell_11_matmul_1_readvariableop_resourceGsequential_6_gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0"
Csequential_6_gru_5_while_gru_cell_11_matmul_readvariableop_resourceEsequential_6_gru_5_while_gru_cell_11_matmul_readvariableop_resource_0"~
<sequential_6_gru_5_while_gru_cell_11_readvariableop_resource>sequential_6_gru_5_while_gru_cell_11_readvariableop_resource_0"O
!sequential_6_gru_5_while_identity*sequential_6/gru_5/while/Identity:output:0"S
#sequential_6_gru_5_while_identity_1,sequential_6/gru_5/while/Identity_1:output:0"S
#sequential_6_gru_5_while_identity_2,sequential_6/gru_5/while/Identity_2:output:0"S
#sequential_6_gru_5_while_identity_3,sequential_6/gru_5/while/Identity_3:output:0"S
#sequential_6_gru_5_while_identity_4,sequential_6/gru_5/while/Identity_4:output:0"|
;sequential_6_gru_5_while_sequential_6_gru_5_strided_slice_1=sequential_6_gru_5_while_sequential_6_gru_5_strided_slice_1_0"ô
wsequential_6_gru_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_5_tensorarrayunstack_tensorlistfromtensorysequential_6_gru_5_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2x
:sequential_6/gru_5/while/gru_cell_11/MatMul/ReadVariableOp:sequential_6/gru_5/while/gru_cell_11/MatMul/ReadVariableOp2|
<sequential_6/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp<sequential_6/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp2j
3sequential_6/gru_5/while/gru_cell_11/ReadVariableOp3sequential_6/gru_5/while/gru_cell_11/ReadVariableOp: 
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

x
#__inference_internal_grad_fn_144043
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
þ

#__inference_internal_grad_fn_143773
result_grads_0
result_grads_1
mul_while_gru_cell_9_beta
mul_while_gru_cell_9_add_2
identity
mulMulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2*
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
ß

#__inference_internal_grad_fn_144115
result_grads_0
result_grads_1
mul_gru_cell_11_beta
mul_gru_cell_11_add_2
identityz
mulMulmul_gru_cell_11_betamul_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_11_betamul_gru_cell_11_add_2*
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
©
¹
&__inference_gru_3_layer_call_fn_140548
inputs_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallò
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_137407|
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
Ú
ª
while_cond_140974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_140974___redundant_placeholder04
0while_while_cond_140974___redundant_placeholder14
0while_while_cond_140974___redundant_placeholder24
0while_while_cond_140974___redundant_placeholder3
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
÷
·
&__inference_gru_5_layer_call_fn_141983

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallã
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138641o
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
¨R

A__inference_gru_5_layer_call_and_return_conditional_losses_138641

inputs6
#gru_cell_11_readvariableop_resource:	¬=
*gru_cell_11_matmul_readvariableop_resource:	d¬?
,gru_cell_11_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_11/MatMul/ReadVariableOp¢#gru_cell_11/MatMul_1/ReadVariableOp¢gru_cell_11/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_11/ReadVariableOpReadVariableOp#gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_11/unstackUnpack"gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_11/MatMul/ReadVariableOpReadVariableOp*gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMulMatMulstrided_slice_2:output:0)gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAddBiasAddgru_cell_11/MatMul:product:0gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_11/splitSplit$gru_cell_11/split/split_dim:output:0gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMul_1MatMulzeros:output:0+gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAdd_1BiasAddgru_cell_11/MatMul_1:product:0gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_11/split_1SplitVgru_cell_11/BiasAdd_1:output:0gru_cell_11/Const:output:0&gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_11/addAddV2gru_cell_11/split:output:0gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_11/SigmoidSigmoidgru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/add_1AddV2gru_cell_11/split:output:1gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_1Sigmoidgru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mulMulgru_cell_11/Sigmoid_1:y:0gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_11/add_2AddV2gru_cell_11/split:output:2gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_11/mul_1Mulgru_cell_11/beta:output:0gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_2Sigmoidgru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_11/mul_2Mulgru_cell_11/add_2:z:0gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/IdentityIdentitygru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_11/IdentityN	IdentityNgru_cell_11/mul_2:z:0gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138529*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_11/mul_3Mulgru_cell_11/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_11/subSubgru_cell_11/sub/x:output:0gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mul_4Mulgru_cell_11/sub:z:0gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_11/add_3AddV2gru_cell_11/mul_3:z:0gru_cell_11/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_11_readvariableop_resource*gru_cell_11_matmul_readvariableop_resource,gru_cell_11_matmul_1_readvariableop_resource*
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
bodyR
while_body_138545*
condR
while_cond_138544*8
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
:ÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_11/MatMul/ReadVariableOp$^gru_cell_11/MatMul_1/ReadVariableOp^gru_cell_11/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_11/MatMul/ReadVariableOp!gru_cell_11/MatMul/ReadVariableOp2J
#gru_cell_11/MatMul_1/ReadVariableOp#gru_cell_11/MatMul_1/ReadVariableOp28
gru_cell_11/ReadVariableOpgru_cell_11/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¨R

A__inference_gru_5_layer_call_and_return_conditional_losses_142495

inputs6
#gru_cell_11_readvariableop_resource:	¬=
*gru_cell_11_matmul_readvariableop_resource:	d¬?
,gru_cell_11_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_11/MatMul/ReadVariableOp¢#gru_cell_11/MatMul_1/ReadVariableOp¢gru_cell_11/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_11/ReadVariableOpReadVariableOp#gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_11/unstackUnpack"gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_11/MatMul/ReadVariableOpReadVariableOp*gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMulMatMulstrided_slice_2:output:0)gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAddBiasAddgru_cell_11/MatMul:product:0gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_11/splitSplit$gru_cell_11/split/split_dim:output:0gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMul_1MatMulzeros:output:0+gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAdd_1BiasAddgru_cell_11/MatMul_1:product:0gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_11/split_1SplitVgru_cell_11/BiasAdd_1:output:0gru_cell_11/Const:output:0&gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_11/addAddV2gru_cell_11/split:output:0gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_11/SigmoidSigmoidgru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/add_1AddV2gru_cell_11/split:output:1gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_1Sigmoidgru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mulMulgru_cell_11/Sigmoid_1:y:0gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_11/add_2AddV2gru_cell_11/split:output:2gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_11/mul_1Mulgru_cell_11/beta:output:0gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_2Sigmoidgru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_11/mul_2Mulgru_cell_11/add_2:z:0gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/IdentityIdentitygru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_11/IdentityN	IdentityNgru_cell_11/mul_2:z:0gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142383*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_11/mul_3Mulgru_cell_11/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_11/subSubgru_cell_11/sub/x:output:0gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mul_4Mulgru_cell_11/sub:z:0gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_11/add_3AddV2gru_cell_11/mul_3:z:0gru_cell_11/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_11_readvariableop_resource*gru_cell_11_matmul_readvariableop_resource,gru_cell_11_matmul_1_readvariableop_resource*
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
bodyR
while_body_142399*
condR
while_cond_142398*8
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
:ÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_11/MatMul/ReadVariableOp$^gru_cell_11/MatMul_1/ReadVariableOp^gru_cell_11/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_11/MatMul/ReadVariableOp!gru_cell_11/MatMul/ReadVariableOp2J
#gru_cell_11/MatMul_1/ReadVariableOp#gru_cell_11/MatMul_1/ReadVariableOp28
gru_cell_11/ReadVariableOpgru_cell_11/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs


#__inference_internal_grad_fn_144169
result_grads_0
result_grads_1
mul_while_gru_cell_11_beta
mul_while_gru_cell_11_add_2
identity
mulMulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2*
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

x
#__inference_internal_grad_fn_144295
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
Ú
ª
while_cond_138370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_138370___redundant_placeholder04
0while_while_cond_138370___redundant_placeholder14
0while_while_cond_138370___redundant_placeholder24
0while_while_cond_138370___redundant_placeholder3
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
è¢
¬
"__inference__traced_restore_144502
file_prefix1
assignvariableop_dense_6_kernel:d-
assignvariableop_1_dense_6_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: =
*assignvariableop_7_gru_3_gru_cell_9_kernel:	¬G
4assignvariableop_8_gru_3_gru_cell_9_recurrent_kernel:	d¬;
(assignvariableop_9_gru_3_gru_cell_9_bias:	¬?
,assignvariableop_10_gru_4_gru_cell_10_kernel:	d¬I
6assignvariableop_11_gru_4_gru_cell_10_recurrent_kernel:	d¬=
*assignvariableop_12_gru_4_gru_cell_10_bias:	¬?
,assignvariableop_13_gru_5_gru_cell_11_kernel:	d¬I
6assignvariableop_14_gru_5_gru_cell_11_recurrent_kernel:	d¬=
*assignvariableop_15_gru_5_gru_cell_11_bias:	¬#
assignvariableop_16_total: #
assignvariableop_17_count: ;
)assignvariableop_18_adam_dense_6_kernel_m:d5
'assignvariableop_19_adam_dense_6_bias_m:E
2assignvariableop_20_adam_gru_3_gru_cell_9_kernel_m:	¬O
<assignvariableop_21_adam_gru_3_gru_cell_9_recurrent_kernel_m:	d¬C
0assignvariableop_22_adam_gru_3_gru_cell_9_bias_m:	¬F
3assignvariableop_23_adam_gru_4_gru_cell_10_kernel_m:	d¬P
=assignvariableop_24_adam_gru_4_gru_cell_10_recurrent_kernel_m:	d¬D
1assignvariableop_25_adam_gru_4_gru_cell_10_bias_m:	¬F
3assignvariableop_26_adam_gru_5_gru_cell_11_kernel_m:	d¬P
=assignvariableop_27_adam_gru_5_gru_cell_11_recurrent_kernel_m:	d¬D
1assignvariableop_28_adam_gru_5_gru_cell_11_bias_m:	¬;
)assignvariableop_29_adam_dense_6_kernel_v:d5
'assignvariableop_30_adam_dense_6_bias_v:E
2assignvariableop_31_adam_gru_3_gru_cell_9_kernel_v:	¬O
<assignvariableop_32_adam_gru_3_gru_cell_9_recurrent_kernel_v:	d¬C
0assignvariableop_33_adam_gru_3_gru_cell_9_bias_v:	¬F
3assignvariableop_34_adam_gru_4_gru_cell_10_kernel_v:	d¬P
=assignvariableop_35_adam_gru_4_gru_cell_10_recurrent_kernel_v:	d¬D
1assignvariableop_36_adam_gru_4_gru_cell_10_bias_v:	¬F
3assignvariableop_37_adam_gru_5_gru_cell_11_kernel_v:	d¬P
=assignvariableop_38_adam_gru_5_gru_cell_11_recurrent_kernel_v:	d¬D
1assignvariableop_39_adam_gru_5_gru_cell_11_bias_v:	¬
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
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp*assignvariableop_7_gru_3_gru_cell_9_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_8AssignVariableOp4assignvariableop_8_gru_3_gru_cell_9_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp(assignvariableop_9_gru_3_gru_cell_9_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp,assignvariableop_10_gru_4_gru_cell_10_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_11AssignVariableOp6assignvariableop_11_gru_4_gru_cell_10_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp*assignvariableop_12_gru_4_gru_cell_10_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp,assignvariableop_13_gru_5_gru_cell_11_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp6assignvariableop_14_gru_5_gru_cell_11_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp*assignvariableop_15_gru_5_gru_cell_11_biasIdentity_15:output:0"/device:CPU:0*
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
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_6_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_6_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_gru_3_gru_cell_9_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_gru_3_gru_cell_9_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_gru_3_gru_cell_9_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_gru_4_gru_cell_10_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_24AssignVariableOp=assignvariableop_24_adam_gru_4_gru_cell_10_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_gru_4_gru_cell_10_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_gru_5_gru_cell_11_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_27AssignVariableOp=assignvariableop_27_adam_gru_5_gru_cell_11_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_gru_5_gru_cell_11_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_6_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_6_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_gru_3_gru_cell_9_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_32AssignVariableOp<assignvariableop_32_adam_gru_3_gru_cell_9_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_gru_3_gru_cell_9_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_gru_4_gru_cell_10_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_35AssignVariableOp=assignvariableop_35_adam_gru_4_gru_cell_10_recurrent_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_gru_4_gru_cell_10_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_gru_5_gru_cell_11_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_38AssignVariableOp=assignvariableop_38_adam_gru_5_gru_cell_11_recurrent_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_gru_5_gru_cell_11_bias_vIdentity_39:output:0"/device:CPU:0*
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

x
#__inference_internal_grad_fn_143683
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
÷B

while_body_138545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_11_readvariableop_resource_0:	¬E
2while_gru_cell_11_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_11_readvariableop_resource:	¬C
0while_gru_cell_11_matmul_readvariableop_resource:	d¬E
2while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_11/MatMul/ReadVariableOp¢)while/gru_cell_11/MatMul_1/ReadVariableOp¢ while/gru_cell_11/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_11/ReadVariableOpReadVariableOp+while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_11/unstackUnpack(while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_11/BiasAddBiasAdd"while/gru_cell_11/MatMul:product:0"while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_11/splitSplit*while/gru_cell_11/split/split_dim:output:0"while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_11/MatMul_1MatMulwhile_placeholder_21while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_11/BiasAdd_1BiasAdd$while/gru_cell_11/MatMul_1:product:0"while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_11/split_1SplitV$while/gru_cell_11/BiasAdd_1:output:0 while/gru_cell_11/Const:output:0,while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_11/addAddV2 while/gru_cell_11/split:output:0"while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_11/SigmoidSigmoidwhile/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_1AddV2 while/gru_cell_11/split:output:1"while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_1Sigmoidwhile/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mulMulwhile/gru_cell_11/Sigmoid_1:y:0"while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_2AddV2 while/gru_cell_11/split:output:2while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/mul_1Mulwhile/gru_cell_11/beta:output:0while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_2Sigmoidwhile/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_2Mulwhile/gru_cell_11/add_2:z:0while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/IdentityIdentitywhile/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_11/IdentityN	IdentityNwhile/gru_cell_11/mul_2:z:0while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138595*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_3Mulwhile/gru_cell_11/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/subSub while/gru_cell_11/sub/x:output:0while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_4Mulwhile/gru_cell_11/sub:z:0$while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_3AddV2while/gru_cell_11/mul_3:z:0while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_11/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_11/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_11/MatMul/ReadVariableOp*^while/gru_cell_11/MatMul_1/ReadVariableOp!^while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_11_matmul_1_readvariableop_resource4while_gru_cell_11_matmul_1_readvariableop_resource_0"f
0while_gru_cell_11_matmul_readvariableop_resource2while_gru_cell_11_matmul_readvariableop_resource_0"X
)while_gru_cell_11_readvariableop_resource+while_gru_cell_11_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_11/MatMul/ReadVariableOp'while/gru_cell_11/MatMul/ReadVariableOp2V
)while/gru_cell_11/MatMul_1/ReadVariableOp)while/gru_cell_11/MatMul_1/ReadVariableOp2D
 while/gru_cell_11/ReadVariableOp while/gru_cell_11/ReadVariableOp: 
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
©
¨
#__inference_internal_grad_fn_143665
result_grads_0
result_grads_1$
 mul_gru_5_while_gru_cell_11_beta%
!mul_gru_5_while_gru_cell_11_add_2
identity
mulMul mul_gru_5_while_gru_cell_11_beta!mul_gru_5_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul mul_gru_5_while_gru_cell_11_beta!mul_gru_5_while_gru_cell_11_add_2*
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
Ú
ª
while_cond_138784
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_138784___redundant_placeholder04
0while_while_cond_138784___redundant_placeholder14
0while_while_cond_138784___redundant_placeholder24
0while_while_cond_138784___redundant_placeholder3
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
K
¼	
gru_5_while_body_140395(
$gru_5_while_gru_5_while_loop_counter.
*gru_5_while_gru_5_while_maximum_iterations
gru_5_while_placeholder
gru_5_while_placeholder_1
gru_5_while_placeholder_2'
#gru_5_while_gru_5_strided_slice_1_0c
_gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_0D
1gru_5_while_gru_cell_11_readvariableop_resource_0:	¬K
8gru_5_while_gru_cell_11_matmul_readvariableop_resource_0:	d¬M
:gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
gru_5_while_identity
gru_5_while_identity_1
gru_5_while_identity_2
gru_5_while_identity_3
gru_5_while_identity_4%
!gru_5_while_gru_5_strided_slice_1a
]gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensorB
/gru_5_while_gru_cell_11_readvariableop_resource:	¬I
6gru_5_while_gru_cell_11_matmul_readvariableop_resource:	d¬K
8gru_5_while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢-gru_5/while/gru_cell_11/MatMul/ReadVariableOp¢/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp¢&gru_5/while/gru_cell_11/ReadVariableOp
=gru_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_0gru_5_while_placeholderFgru_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
&gru_5/while/gru_cell_11/ReadVariableOpReadVariableOp1gru_5_while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_5/while/gru_cell_11/unstackUnpack.gru_5/while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num§
-gru_5/while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp8gru_5_while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ê
gru_5/while/gru_cell_11/MatMulMatMul6gru_5/while/TensorArrayV2Read/TensorListGetItem:item:05gru_5/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
gru_5/while/gru_cell_11/BiasAddBiasAdd(gru_5/while/gru_cell_11/MatMul:product:0(gru_5/while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
'gru_5/while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_5/while/gru_cell_11/splitSplit0gru_5/while/gru_cell_11/split/split_dim:output:0(gru_5/while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split«
/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp:gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0±
 gru_5/while/gru_cell_11/MatMul_1MatMulgru_5_while_placeholder_27gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬µ
!gru_5/while/gru_cell_11/BiasAdd_1BiasAdd*gru_5/while/gru_cell_11/MatMul_1:product:0(gru_5/while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
gru_5/while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿt
)gru_5/while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_5/while/gru_cell_11/split_1SplitV*gru_5/while/gru_cell_11/BiasAdd_1:output:0&gru_5/while/gru_cell_11/Const:output:02gru_5/while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¨
gru_5/while/gru_cell_11/addAddV2&gru_5/while/gru_cell_11/split:output:0(gru_5/while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_5/while/gru_cell_11/SigmoidSigmoidgru_5/while/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
gru_5/while/gru_cell_11/add_1AddV2&gru_5/while/gru_cell_11/split:output:1(gru_5/while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_5/while/gru_cell_11/Sigmoid_1Sigmoid!gru_5/while/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
gru_5/while/gru_cell_11/mulMul%gru_5/while/gru_cell_11/Sigmoid_1:y:0(gru_5/while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¡
gru_5/while/gru_cell_11/add_2AddV2&gru_5/while/gru_cell_11/split:output:2gru_5/while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_5/while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ? 
gru_5/while/gru_cell_11/mul_1Mul%gru_5/while/gru_cell_11/beta:output:0!gru_5/while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_5/while/gru_cell_11/Sigmoid_2Sigmoid!gru_5/while/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_5/while/gru_cell_11/mul_2Mul!gru_5/while/gru_cell_11/add_2:z:0%gru_5/while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_5/while/gru_cell_11/IdentityIdentity!gru_5/while/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdë
!gru_5/while/gru_cell_11/IdentityN	IdentityN!gru_5/while/gru_cell_11/mul_2:z:0!gru_5/while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140445*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_5/while/gru_cell_11/mul_3Mul#gru_5/while/gru_cell_11/Sigmoid:y:0gru_5_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
gru_5/while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_5/while/gru_cell_11/subSub&gru_5/while/gru_cell_11/sub/x:output:0#gru_5/while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd£
gru_5/while/gru_cell_11/mul_4Mulgru_5/while/gru_cell_11/sub:z:0*gru_5/while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/while/gru_cell_11/add_3AddV2!gru_5/while/gru_cell_11/mul_3:z:0!gru_5/while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÜ
0gru_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_5_while_placeholder_1gru_5_while_placeholder!gru_5/while/gru_cell_11/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_5/while/addAddV2gru_5_while_placeholdergru_5/while/add/y:output:0*
T0*
_output_shapes
: U
gru_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_5/while/add_1AddV2$gru_5_while_gru_5_while_loop_countergru_5/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_5/while/IdentityIdentitygru_5/while/add_1:z:0^gru_5/while/NoOp*
T0*
_output_shapes
: 
gru_5/while/Identity_1Identity*gru_5_while_gru_5_while_maximum_iterations^gru_5/while/NoOp*
T0*
_output_shapes
: k
gru_5/while/Identity_2Identitygru_5/while/add:z:0^gru_5/while/NoOp*
T0*
_output_shapes
: «
gru_5/while/Identity_3Identity@gru_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_5/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_5/while/Identity_4Identity!gru_5/while/gru_cell_11/add_3:z:0^gru_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÝ
gru_5/while/NoOpNoOp.^gru_5/while/gru_cell_11/MatMul/ReadVariableOp0^gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp'^gru_5/while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_5_while_gru_5_strided_slice_1#gru_5_while_gru_5_strided_slice_1_0"v
8gru_5_while_gru_cell_11_matmul_1_readvariableop_resource:gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0"r
6gru_5_while_gru_cell_11_matmul_readvariableop_resource8gru_5_while_gru_cell_11_matmul_readvariableop_resource_0"d
/gru_5_while_gru_cell_11_readvariableop_resource1gru_5_while_gru_cell_11_readvariableop_resource_0"5
gru_5_while_identitygru_5/while/Identity:output:0"9
gru_5_while_identity_1gru_5/while/Identity_1:output:0"9
gru_5_while_identity_2gru_5/while/Identity_2:output:0"9
gru_5_while_identity_3gru_5/while/Identity_3:output:0"9
gru_5_while_identity_4gru_5/while/Identity_4:output:0"À
]gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2^
-gru_5/while/gru_cell_11/MatMul/ReadVariableOp-gru_5/while/gru_cell_11/MatMul/ReadVariableOp2b
/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp2P
&gru_5/while/gru_cell_11/ReadVariableOp&gru_5/while/gru_cell_11/ReadVariableOp: 
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
þ

#__inference_internal_grad_fn_143845
result_grads_0
result_grads_1
mul_while_gru_cell_9_beta
mul_while_gru_cell_9_add_2
identity
mulMulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2*
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
!
Ú
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137291

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137277*:
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
÷
Â
#__inference_internal_grad_fn_143215
result_grads_0
result_grads_11
-mul_sequential_6_gru_4_while_gru_cell_10_beta2
.mul_sequential_6_gru_4_while_gru_cell_10_add_2
identity¬
mulMul-mul_sequential_6_gru_4_while_gru_cell_10_beta.mul_sequential_6_gru_4_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul-mul_sequential_6_gru_4_while_gru_cell_10_beta.mul_sequential_6_gru_4_while_gru_cell_10_add_2*
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
ü

gru_5_while_cond_139895(
$gru_5_while_gru_5_while_loop_counter.
*gru_5_while_gru_5_while_maximum_iterations
gru_5_while_placeholder
gru_5_while_placeholder_1
gru_5_while_placeholder_2*
&gru_5_while_less_gru_5_strided_slice_1@
<gru_5_while_gru_5_while_cond_139895___redundant_placeholder0@
<gru_5_while_gru_5_while_cond_139895___redundant_placeholder1@
<gru_5_while_gru_5_while_cond_139895___redundant_placeholder2@
<gru_5_while_gru_5_while_cond_139895___redundant_placeholder3
gru_5_while_identity
z
gru_5/while/LessLessgru_5_while_placeholder&gru_5_while_less_gru_5_strided_slice_1*
T0*
_output_shapes
: W
gru_5/while/IdentityIdentitygru_5/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_5_while_identitygru_5/while/Identity:output:0*(
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
À

(__inference_dense_6_layer_call_fn_142671

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallØ
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
GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_138659o
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
´

Û
,__inference_gru_cell_10_layer_call_fn_142829

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137643o
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
!
Ý
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142921

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142907*:
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
Ú
ª
while_cond_137505
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_137505___redundant_placeholder04
0while_while_cond_137505___redundant_placeholder14
0while_while_cond_137505___redundant_placeholder24
0while_while_cond_137505___redundant_placeholder3
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
Ú
ª
while_cond_142064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_142064___redundant_placeholder04
0while_while_cond_142064___redundant_placeholder14
0while_while_cond_142064___redundant_placeholder24
0while_while_cond_142064___redundant_placeholder3
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


#__inference_internal_grad_fn_143989
result_grads_0
result_grads_1
mul_while_gru_cell_10_beta
mul_while_gru_cell_10_add_2
identity
mulMulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2*
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
¹
Ò
H__inference_sequential_6_layer_call_and_return_conditional_losses_138666

inputs
gru_3_138294:	¬
gru_3_138296:	¬
gru_3_138298:	d¬
gru_4_138468:	¬
gru_4_138470:	d¬
gru_4_138472:	d¬
gru_5_138642:	¬
gru_5_138644:	d¬
gru_5_138646:	d¬ 
dense_6_138660:d
dense_6_138662:
identity¢dense_6/StatefulPartitionedCall¢gru_3/StatefulPartitionedCall¢gru_4/StatefulPartitionedCall¢gru_5/StatefulPartitionedCallø
gru_3/StatefulPartitionedCallStatefulPartitionedCallinputsgru_3_138294gru_3_138296gru_3_138298*
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_138293
gru_4/StatefulPartitionedCallStatefulPartitionedCall&gru_3/StatefulPartitionedCall:output:0gru_4_138468gru_4_138470gru_4_138472*
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_138467
gru_5/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0gru_5_138642gru_5_138644gru_5_138646*
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138641
dense_6/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0dense_6_138660dense_6_138662*
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
GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_138659w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_6/StatefulPartitionedCall^gru_3/StatefulPartitionedCall^gru_4/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2>
gru_3/StatefulPartitionedCallgru_3/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ù

#__inference_internal_grad_fn_143827
result_grads_0
result_grads_1
mul_gru_cell_9_beta
mul_gru_cell_9_add_2
identityx
mulMulmul_gru_cell_9_betamul_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_9_betamul_gru_cell_9_add_2*
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
ü

gru_4_while_cond_140231(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2*
&gru_4_while_less_gru_4_strided_slice_1@
<gru_4_while_gru_4_while_cond_140231___redundant_placeholder0@
<gru_4_while_gru_4_while_cond_140231___redundant_placeholder1@
<gru_4_while_gru_4_while_cond_140231___redundant_placeholder2@
<gru_4_while_gru_4_while_cond_140231___redundant_placeholder3
gru_4_while_identity
z
gru_4/while/LessLessgru_4_while_placeholder&gru_4_while_less_gru_4_strided_slice_1*
T0*
_output_shapes
: W
gru_4/while/IdentityIdentitygru_4/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_4_while_identitygru_4/while/Identity:output:0*(
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
!
Û
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137493

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137479*:
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
ß

#__inference_internal_grad_fn_144187
result_grads_0
result_grads_1
mul_gru_cell_11_beta
mul_gru_cell_11_add_2
identityz
mulMulmul_gru_cell_11_betamul_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_11_betamul_gru_cell_11_add_2*
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
4

A__inference_gru_5_layer_call_and_return_conditional_losses_137922

inputs%
gru_cell_11_137846:	¬%
gru_cell_11_137848:	d¬%
gru_cell_11_137850:	d¬
identity¢#gru_cell_11/StatefulPartitionedCall¢while;
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
shrink_axis_maskÉ
#gru_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_11_137846gru_cell_11_137848gru_cell_11_137850*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137845n
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
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_11_137846gru_cell_11_137848gru_cell_11_137850*
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
bodyR
while_body_137858*
condR
while_cond_137857*8
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
:ÿÿÿÿÿÿÿÿÿdt
NoOpNoOp$^gru_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2J
#gru_cell_11/StatefulPartitionedCall#gru_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ß

#__inference_internal_grad_fn_144007
result_grads_0
result_grads_1
mul_gru_cell_10_beta
mul_gru_cell_10_add_2
identityz
mulMulmul_gru_cell_10_betamul_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_10_betamul_gru_cell_10_add_2*
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


#__inference_internal_grad_fn_144205
result_grads_0
result_grads_1
mul_while_gru_cell_11_beta
mul_while_gru_cell_11_add_2
identity
mulMulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2*
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


#__inference_internal_grad_fn_143341
result_grads_0
result_grads_1
mul_while_gru_cell_11_beta
mul_while_gru_cell_11_add_2
identity
mulMulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2*
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
È
×
H__inference_sequential_6_layer_call_and_return_conditional_losses_139439
gru_3_input
gru_3_139412:	¬
gru_3_139414:	¬
gru_3_139416:	d¬
gru_4_139419:	¬
gru_4_139421:	d¬
gru_4_139423:	d¬
gru_5_139426:	¬
gru_5_139428:	d¬
gru_5_139430:	d¬ 
dense_6_139433:d
dense_6_139435:
identity¢dense_6/StatefulPartitionedCall¢gru_3/StatefulPartitionedCall¢gru_4/StatefulPartitionedCall¢gru_5/StatefulPartitionedCallý
gru_3/StatefulPartitionedCallStatefulPartitionedCallgru_3_inputgru_3_139412gru_3_139414gru_3_139416*
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_139259
gru_4/StatefulPartitionedCallStatefulPartitionedCall&gru_3/StatefulPartitionedCall:output:0gru_4_139419gru_4_139421gru_4_139423*
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_139070
gru_5/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0gru_5_139426gru_5_139428gru_5_139430*
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138881
dense_6/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0dense_6_139433dense_6_139435*
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
GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_138659w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_6/StatefulPartitionedCall^gru_3/StatefulPartitionedCall^gru_4/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2>
gru_3/StatefulPartitionedCallgru_3/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_3_input

x
#__inference_internal_grad_fn_144313
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
!
Ú
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137141

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137127*:
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
Ù

#__inference_internal_grad_fn_143359
result_grads_0
result_grads_1
mul_gru_cell_9_beta
mul_gru_cell_9_add_2
identityx
mulMulmul_gru_cell_9_betamul_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_9_betamul_gru_cell_9_add_2*
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
ÝR

A__inference_gru_5_layer_call_and_return_conditional_losses_142161
inputs_06
#gru_cell_11_readvariableop_resource:	¬=
*gru_cell_11_matmul_readvariableop_resource:	d¬?
,gru_cell_11_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_11/MatMul/ReadVariableOp¢#gru_cell_11/MatMul_1/ReadVariableOp¢gru_cell_11/ReadVariableOp¢while=
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
shrink_axis_mask
gru_cell_11/ReadVariableOpReadVariableOp#gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_11/unstackUnpack"gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_11/MatMul/ReadVariableOpReadVariableOp*gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMulMatMulstrided_slice_2:output:0)gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAddBiasAddgru_cell_11/MatMul:product:0gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_11/splitSplit$gru_cell_11/split/split_dim:output:0gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMul_1MatMulzeros:output:0+gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAdd_1BiasAddgru_cell_11/MatMul_1:product:0gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_11/split_1SplitVgru_cell_11/BiasAdd_1:output:0gru_cell_11/Const:output:0&gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_11/addAddV2gru_cell_11/split:output:0gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_11/SigmoidSigmoidgru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/add_1AddV2gru_cell_11/split:output:1gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_1Sigmoidgru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mulMulgru_cell_11/Sigmoid_1:y:0gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_11/add_2AddV2gru_cell_11/split:output:2gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_11/mul_1Mulgru_cell_11/beta:output:0gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_2Sigmoidgru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_11/mul_2Mulgru_cell_11/add_2:z:0gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/IdentityIdentitygru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_11/IdentityN	IdentityNgru_cell_11/mul_2:z:0gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142049*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_11/mul_3Mulgru_cell_11/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_11/subSubgru_cell_11/sub/x:output:0gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mul_4Mulgru_cell_11/sub:z:0gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_11/add_3AddV2gru_cell_11/mul_3:z:0gru_cell_11/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_11_readvariableop_resource*gru_cell_11_matmul_readvariableop_resource,gru_cell_11_matmul_1_readvariableop_resource*
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
bodyR
while_body_142065*
condR
while_cond_142064*8
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
:ÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_11/MatMul/ReadVariableOp$^gru_cell_11/MatMul_1/ReadVariableOp^gru_cell_11/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2F
!gru_cell_11/MatMul/ReadVariableOp!gru_cell_11/MatMul/ReadVariableOp2J
#gru_cell_11/MatMul_1/ReadVariableOp#gru_cell_11/MatMul_1/ReadVariableOp28
gru_cell_11/ReadVariableOpgru_cell_11/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
þ

#__inference_internal_grad_fn_143377
result_grads_0
result_grads_1
mul_while_gru_cell_9_beta
mul_while_gru_cell_9_add_2
identity
mulMulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2*
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
÷B

while_body_142566
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_11_readvariableop_resource_0:	¬E
2while_gru_cell_11_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_11_readvariableop_resource:	¬C
0while_gru_cell_11_matmul_readvariableop_resource:	d¬E
2while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_11/MatMul/ReadVariableOp¢)while/gru_cell_11/MatMul_1/ReadVariableOp¢ while/gru_cell_11/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_11/ReadVariableOpReadVariableOp+while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_11/unstackUnpack(while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_11/BiasAddBiasAdd"while/gru_cell_11/MatMul:product:0"while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_11/splitSplit*while/gru_cell_11/split/split_dim:output:0"while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_11/MatMul_1MatMulwhile_placeholder_21while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_11/BiasAdd_1BiasAdd$while/gru_cell_11/MatMul_1:product:0"while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_11/split_1SplitV$while/gru_cell_11/BiasAdd_1:output:0 while/gru_cell_11/Const:output:0,while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_11/addAddV2 while/gru_cell_11/split:output:0"while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_11/SigmoidSigmoidwhile/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_1AddV2 while/gru_cell_11/split:output:1"while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_1Sigmoidwhile/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mulMulwhile/gru_cell_11/Sigmoid_1:y:0"while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_2AddV2 while/gru_cell_11/split:output:2while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/mul_1Mulwhile/gru_cell_11/beta:output:0while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_2Sigmoidwhile/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_2Mulwhile/gru_cell_11/add_2:z:0while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/IdentityIdentitywhile/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_11/IdentityN	IdentityNwhile/gru_cell_11/mul_2:z:0while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142616*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_3Mulwhile/gru_cell_11/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/subSub while/gru_cell_11/sub/x:output:0while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_4Mulwhile/gru_cell_11/sub:z:0$while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_3AddV2while/gru_cell_11/mul_3:z:0while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_11/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_11/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_11/MatMul/ReadVariableOp*^while/gru_cell_11/MatMul_1/ReadVariableOp!^while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_11_matmul_1_readvariableop_resource4while_gru_cell_11_matmul_1_readvariableop_resource_0"f
0while_gru_cell_11_matmul_readvariableop_resource2while_gru_cell_11_matmul_readvariableop_resource_0"X
)while_gru_cell_11_readvariableop_resource+while_gru_cell_11_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_11/MatMul/ReadVariableOp'while/gru_cell_11/MatMul/ReadVariableOp2V
)while/gru_cell_11/MatMul_1/ReadVariableOp)while/gru_cell_11/MatMul_1/ReadVariableOp2D
 while/gru_cell_11/ReadVariableOp while/gru_cell_11/ReadVariableOp: 
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
þ

#__inference_internal_grad_fn_143737
result_grads_0
result_grads_1
mul_while_gru_cell_9_beta
mul_while_gru_cell_9_add_2
identity
mulMulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2*
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
4

A__inference_gru_5_layer_call_and_return_conditional_losses_138111

inputs%
gru_cell_11_138035:	¬%
gru_cell_11_138037:	d¬%
gru_cell_11_138039:	d¬
identity¢#gru_cell_11/StatefulPartitionedCall¢while;
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
shrink_axis_maskÉ
#gru_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_11_138035gru_cell_11_138037gru_cell_11_138039*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137995n
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
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_11_138035gru_cell_11_138037gru_cell_11_138039*
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
bodyR
while_body_138047*
condR
while_cond_138046*8
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
:ÿÿÿÿÿÿÿÿÿdt
NoOpNoOp$^gru_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2J
#gru_cell_11/StatefulPartitionedCall#gru_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


#__inference_internal_grad_fn_144025
result_grads_0
result_grads_1
mul_while_gru_cell_10_beta
mul_while_gru_cell_10_add_2
identity
mulMulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2*
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
£
¦
#__inference_internal_grad_fn_143521
result_grads_0
result_grads_1#
mul_gru_3_while_gru_cell_9_beta$
 mul_gru_3_while_gru_cell_9_add_2
identity
mulMulmul_gru_3_while_gru_cell_9_beta mul_gru_3_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_3_while_gru_cell_9_beta mul_gru_3_while_gru_cell_9_add_2*
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
´

Û
,__inference_gru_cell_11_layer_call_fn_142935

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137845o
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
áR

A__inference_gru_4_layer_call_and_return_conditional_losses_141449
inputs_06
#gru_cell_10_readvariableop_resource:	¬=
*gru_cell_10_matmul_readvariableop_resource:	d¬?
,gru_cell_10_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_10/MatMul/ReadVariableOp¢#gru_cell_10/MatMul_1/ReadVariableOp¢gru_cell_10/ReadVariableOp¢while=
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
shrink_axis_mask
gru_cell_10/ReadVariableOpReadVariableOp#gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_10/unstackUnpack"gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_10/MatMul/ReadVariableOpReadVariableOp*gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMulMatMulstrided_slice_2:output:0)gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAddBiasAddgru_cell_10/MatMul:product:0gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_10/splitSplit$gru_cell_10/split/split_dim:output:0gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMul_1MatMulzeros:output:0+gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAdd_1BiasAddgru_cell_10/MatMul_1:product:0gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_10/split_1SplitVgru_cell_10/BiasAdd_1:output:0gru_cell_10/Const:output:0&gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_10/addAddV2gru_cell_10/split:output:0gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_10/SigmoidSigmoidgru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/add_1AddV2gru_cell_10/split:output:1gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_1Sigmoidgru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mulMulgru_cell_10/Sigmoid_1:y:0gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_10/add_2AddV2gru_cell_10/split:output:2gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_10/mul_1Mulgru_cell_10/beta:output:0gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_2Sigmoidgru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_10/mul_2Mulgru_cell_10/add_2:z:0gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/IdentityIdentitygru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_10/IdentityN	IdentityNgru_cell_10/mul_2:z:0gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141337*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_10/mul_3Mulgru_cell_10/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_10/subSubgru_cell_10/sub/x:output:0gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mul_4Mulgru_cell_10/sub:z:0gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_10/add_3AddV2gru_cell_10/mul_3:z:0gru_cell_10/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_10_readvariableop_resource*gru_cell_10_matmul_readvariableop_resource,gru_cell_10_matmul_1_readvariableop_resource*
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
bodyR
while_body_141353*
condR
while_cond_141352*8
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_10/MatMul/ReadVariableOp$^gru_cell_10/MatMul_1/ReadVariableOp^gru_cell_10/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2F
!gru_cell_10/MatMul/ReadVariableOp!gru_cell_10/MatMul/ReadVariableOp2J
#gru_cell_10/MatMul_1/ReadVariableOp#gru_cell_10/MatMul_1/ReadVariableOp28
gru_cell_10/ReadVariableOpgru_cell_10/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
Ù

#__inference_internal_grad_fn_143791
result_grads_0
result_grads_1
mul_gru_cell_9_beta
mul_gru_cell_9_add_2
identityx
mulMulmul_gru_cell_9_betamul_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_9_betamul_gru_cell_9_add_2*
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
Ü

$sequential_6_gru_3_while_cond_136635B
>sequential_6_gru_3_while_sequential_6_gru_3_while_loop_counterH
Dsequential_6_gru_3_while_sequential_6_gru_3_while_maximum_iterations(
$sequential_6_gru_3_while_placeholder*
&sequential_6_gru_3_while_placeholder_1*
&sequential_6_gru_3_while_placeholder_2D
@sequential_6_gru_3_while_less_sequential_6_gru_3_strided_slice_1Z
Vsequential_6_gru_3_while_sequential_6_gru_3_while_cond_136635___redundant_placeholder0Z
Vsequential_6_gru_3_while_sequential_6_gru_3_while_cond_136635___redundant_placeholder1Z
Vsequential_6_gru_3_while_sequential_6_gru_3_while_cond_136635___redundant_placeholder2Z
Vsequential_6_gru_3_while_sequential_6_gru_3_while_cond_136635___redundant_placeholder3%
!sequential_6_gru_3_while_identity
®
sequential_6/gru_3/while/LessLess$sequential_6_gru_3_while_placeholder@sequential_6_gru_3_while_less_sequential_6_gru_3_strided_slice_1*
T0*
_output_shapes
: q
!sequential_6/gru_3/while/IdentityIdentity!sequential_6/gru_3/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_6_gru_3_while_identity*sequential_6/gru_3/while/Identity:output:0*(
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
ñ
À
#__inference_internal_grad_fn_143197
result_grads_0
result_grads_10
,mul_sequential_6_gru_3_while_gru_cell_9_beta1
-mul_sequential_6_gru_3_while_gru_cell_9_add_2
identityª
mulMul,mul_sequential_6_gru_3_while_gru_cell_9_beta-mul_sequential_6_gru_3_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul,mul_sequential_6_gru_3_while_gru_cell_9_beta-mul_sequential_6_gru_3_while_gru_cell_9_add_2*
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
ö

­
-__inference_sequential_6_layer_call_fn_138691
gru_3_input
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
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallgru_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
GPU 2J 8 *Q
fLRJ
H__inference_sequential_6_layer_call_and_return_conditional_losses_138666o
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
_user_specified_namegru_3_input
ÊQ

A__inference_gru_3_layer_call_and_return_conditional_losses_139259

inputs5
"gru_cell_9_readvariableop_resource:	¬<
)gru_cell_9_matmul_readvariableop_resource:	¬>
+gru_cell_9_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_9/MatMul/ReadVariableOp¢"gru_cell_9/MatMul_1/ReadVariableOp¢gru_cell_9/ReadVariableOp¢while;
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
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_9/mul_1Mulgru_cell_9/beta:output:0gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_2Sigmoidgru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_9/mul_2Mulgru_cell_9/add_2:z:0gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/IdentityIdentitygru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
gru_cell_9/IdentityN	IdentityNgru_cell_9/mul_2:z:0gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139147*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_9/mul_3Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_9/mul_4Mulgru_cell_9/sub:z:0gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_9/add_3AddV2gru_cell_9/mul_3:z:0gru_cell_9/mul_4:z:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
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
bodyR
while_body_139163*
condR
while_cond_139162*8
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
NoOpNoOp!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ú
ª
while_cond_138046
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_138046___redundant_placeholder04
0while_while_cond_138046___redundant_placeholder14
0while_while_cond_138046___redundant_placeholder24
0while_while_cond_138046___redundant_placeholder3
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
ß

#__inference_internal_grad_fn_143971
result_grads_0
result_grads_1
mul_gru_cell_10_beta
mul_gru_cell_10_add_2
identityz
mulMulmul_gru_cell_10_betamul_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_10_betamul_gru_cell_10_add_2*
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
÷
Â
#__inference_internal_grad_fn_143233
result_grads_0
result_grads_11
-mul_sequential_6_gru_5_while_gru_cell_11_beta2
.mul_sequential_6_gru_5_while_gru_cell_11_add_2
identity¬
mulMul-mul_sequential_6_gru_5_while_gru_cell_11_beta.mul_sequential_6_gru_5_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul-mul_sequential_6_gru_5_while_gru_cell_11_beta.mul_sequential_6_gru_5_while_gru_cell_11_add_2*
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
¡J
³	
gru_3_while_body_139570(
$gru_3_while_gru_3_while_loop_counter.
*gru_3_while_gru_3_while_maximum_iterations
gru_3_while_placeholder
gru_3_while_placeholder_1
gru_3_while_placeholder_2'
#gru_3_while_gru_3_strided_slice_1_0c
_gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_0C
0gru_3_while_gru_cell_9_readvariableop_resource_0:	¬J
7gru_3_while_gru_cell_9_matmul_readvariableop_resource_0:	¬L
9gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
gru_3_while_identity
gru_3_while_identity_1
gru_3_while_identity_2
gru_3_while_identity_3
gru_3_while_identity_4%
!gru_3_while_gru_3_strided_slice_1a
]gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensorA
.gru_3_while_gru_cell_9_readvariableop_resource:	¬H
5gru_3_while_gru_cell_9_matmul_readvariableop_resource:	¬J
7gru_3_while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢,gru_3/while/gru_cell_9/MatMul/ReadVariableOp¢.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp¢%gru_3/while/gru_cell_9/ReadVariableOp
=gru_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/gru_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_0gru_3_while_placeholderFgru_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%gru_3/while/gru_cell_9/ReadVariableOpReadVariableOp0gru_3_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_3/while/gru_cell_9/unstackUnpack-gru_3/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_3/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7gru_3_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0È
gru_3/while/gru_cell_9/MatMulMatMul6gru_3/while/TensorArrayV2Read/TensorListGetItem:item:04gru_3/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_3/while/gru_cell_9/BiasAddBiasAdd'gru_3/while/gru_cell_9/MatMul:product:0'gru_3/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_3/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_3/while/gru_cell_9/splitSplit/gru_3/while/gru_cell_9/split/split_dim:output:0'gru_3/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_3/while/gru_cell_9/MatMul_1MatMulgru_3_while_placeholder_26gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_3/while/gru_cell_9/BiasAdd_1BiasAdd)gru_3/while/gru_cell_9/MatMul_1:product:0'gru_3/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_3/while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_3/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_3/while/gru_cell_9/split_1SplitV)gru_3/while/gru_cell_9/BiasAdd_1:output:0%gru_3/while/gru_cell_9/Const:output:01gru_3/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_3/while/gru_cell_9/addAddV2%gru_3/while/gru_cell_9/split:output:0'gru_3/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_3/while/gru_cell_9/SigmoidSigmoidgru_3/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_3/while/gru_cell_9/add_1AddV2%gru_3/while/gru_cell_9/split:output:1'gru_3/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_3/while/gru_cell_9/Sigmoid_1Sigmoid gru_3/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_3/while/gru_cell_9/mulMul$gru_3/while/gru_cell_9/Sigmoid_1:y:0'gru_3/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/add_2AddV2%gru_3/while/gru_cell_9/split:output:2gru_3/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_3/while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/while/gru_cell_9/mul_1Mul$gru_3/while/gru_cell_9/beta:output:0 gru_3/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_3/while/gru_cell_9/Sigmoid_2Sigmoid gru_3/while/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/mul_2Mul gru_3/while/gru_cell_9/add_2:z:0$gru_3/while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/IdentityIdentity gru_3/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdè
 gru_3/while/gru_cell_9/IdentityN	IdentityN gru_3/while/gru_cell_9/mul_2:z:0 gru_3/while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139620*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/mul_3Mul"gru_3/while/gru_cell_9/Sigmoid:y:0gru_3_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_3/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/while/gru_cell_9/subSub%gru_3/while/gru_cell_9/sub/x:output:0"gru_3/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_3/while/gru_cell_9/mul_4Mulgru_3/while/gru_cell_9/sub:z:0)gru_3/while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/add_3AddV2 gru_3/while/gru_cell_9/mul_3:z:0 gru_3/while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_3_while_placeholder_1gru_3_while_placeholder gru_3/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_3/while/addAddV2gru_3_while_placeholdergru_3/while/add/y:output:0*
T0*
_output_shapes
: U
gru_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_3/while/add_1AddV2$gru_3_while_gru_3_while_loop_countergru_3/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_3/while/IdentityIdentitygru_3/while/add_1:z:0^gru_3/while/NoOp*
T0*
_output_shapes
: 
gru_3/while/Identity_1Identity*gru_3_while_gru_3_while_maximum_iterations^gru_3/while/NoOp*
T0*
_output_shapes
: k
gru_3/while/Identity_2Identitygru_3/while/add:z:0^gru_3/while/NoOp*
T0*
_output_shapes
: «
gru_3/while/Identity_3Identity@gru_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_3/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_3/while/Identity_4Identity gru_3/while/gru_cell_9/add_3:z:0^gru_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_3/while/NoOpNoOp-^gru_3/while/gru_cell_9/MatMul/ReadVariableOp/^gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_3/while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_3_while_gru_3_strided_slice_1#gru_3_while_gru_3_strided_slice_1_0"t
7gru_3_while_gru_cell_9_matmul_1_readvariableop_resource9gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0"p
5gru_3_while_gru_cell_9_matmul_readvariableop_resource7gru_3_while_gru_cell_9_matmul_readvariableop_resource_0"b
.gru_3_while_gru_cell_9_readvariableop_resource0gru_3_while_gru_cell_9_readvariableop_resource_0"5
gru_3_while_identitygru_3/while/Identity:output:0"9
gru_3_while_identity_1gru_3/while/Identity_1:output:0"9
gru_3_while_identity_2gru_3/while/Identity_2:output:0"9
gru_3_while_identity_3gru_3/while/Identity_3:output:0"9
gru_3_while_identity_4gru_3/while/Identity_4:output:0"À
]gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_3/while/gru_cell_9/MatMul/ReadVariableOp,gru_3/while/gru_cell_9/MatMul/ReadVariableOp2`
.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp2N
%gru_3/while/gru_cell_9/ReadVariableOp%gru_3/while/gru_cell_9/ReadVariableOp: 
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
¨R

A__inference_gru_5_layer_call_and_return_conditional_losses_142662

inputs6
#gru_cell_11_readvariableop_resource:	¬=
*gru_cell_11_matmul_readvariableop_resource:	d¬?
,gru_cell_11_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_11/MatMul/ReadVariableOp¢#gru_cell_11/MatMul_1/ReadVariableOp¢gru_cell_11/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_11/ReadVariableOpReadVariableOp#gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_11/unstackUnpack"gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_11/MatMul/ReadVariableOpReadVariableOp*gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMulMatMulstrided_slice_2:output:0)gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAddBiasAddgru_cell_11/MatMul:product:0gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_11/splitSplit$gru_cell_11/split/split_dim:output:0gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMul_1MatMulzeros:output:0+gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAdd_1BiasAddgru_cell_11/MatMul_1:product:0gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_11/split_1SplitVgru_cell_11/BiasAdd_1:output:0gru_cell_11/Const:output:0&gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_11/addAddV2gru_cell_11/split:output:0gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_11/SigmoidSigmoidgru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/add_1AddV2gru_cell_11/split:output:1gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_1Sigmoidgru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mulMulgru_cell_11/Sigmoid_1:y:0gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_11/add_2AddV2gru_cell_11/split:output:2gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_11/mul_1Mulgru_cell_11/beta:output:0gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_2Sigmoidgru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_11/mul_2Mulgru_cell_11/add_2:z:0gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/IdentityIdentitygru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_11/IdentityN	IdentityNgru_cell_11/mul_2:z:0gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142550*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_11/mul_3Mulgru_cell_11/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_11/subSubgru_cell_11/sub/x:output:0gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mul_4Mulgru_cell_11/sub:z:0gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_11/add_3AddV2gru_cell_11/mul_3:z:0gru_cell_11/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_11_readvariableop_resource*gru_cell_11_matmul_readvariableop_resource,gru_cell_11_matmul_1_readvariableop_resource*
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
bodyR
while_body_142566*
condR
while_cond_142565*8
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
:ÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_11/MatMul/ReadVariableOp$^gru_cell_11/MatMul_1/ReadVariableOp^gru_cell_11/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_11/MatMul/ReadVariableOp!gru_cell_11/MatMul/ReadVariableOp2J
#gru_cell_11/MatMul_1/ReadVariableOp#gru_cell_11/MatMul_1/ReadVariableOp28
gru_cell_11/ReadVariableOpgru_cell_11/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ú
ª
while_cond_142565
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_142565___redundant_placeholder04
0while_while_cond_142565___redundant_placeholder14
0while_while_cond_142565___redundant_placeholder24
0while_while_cond_142565___redundant_placeholder3
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
ß

#__inference_internal_grad_fn_143935
result_grads_0
result_grads_1
mul_gru_cell_10_beta
mul_gru_cell_10_add_2
identityz
mulMulmul_gru_cell_10_betamul_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_10_betamul_gru_cell_10_add_2*
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


#__inference_internal_grad_fn_143503
result_grads_0
result_grads_1
mul_gru_5_gru_cell_11_beta
mul_gru_5_gru_cell_11_add_2
identity
mulMulmul_gru_5_gru_cell_11_betamul_gru_5_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_gru_5_gru_cell_11_betamul_gru_5_gru_cell_11_add_2*
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
K
¼	
gru_5_while_body_139896(
$gru_5_while_gru_5_while_loop_counter.
*gru_5_while_gru_5_while_maximum_iterations
gru_5_while_placeholder
gru_5_while_placeholder_1
gru_5_while_placeholder_2'
#gru_5_while_gru_5_strided_slice_1_0c
_gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_0D
1gru_5_while_gru_cell_11_readvariableop_resource_0:	¬K
8gru_5_while_gru_cell_11_matmul_readvariableop_resource_0:	d¬M
:gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
gru_5_while_identity
gru_5_while_identity_1
gru_5_while_identity_2
gru_5_while_identity_3
gru_5_while_identity_4%
!gru_5_while_gru_5_strided_slice_1a
]gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensorB
/gru_5_while_gru_cell_11_readvariableop_resource:	¬I
6gru_5_while_gru_cell_11_matmul_readvariableop_resource:	d¬K
8gru_5_while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢-gru_5/while/gru_cell_11/MatMul/ReadVariableOp¢/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp¢&gru_5/while/gru_cell_11/ReadVariableOp
=gru_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_0gru_5_while_placeholderFgru_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
&gru_5/while/gru_cell_11/ReadVariableOpReadVariableOp1gru_5_while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_5/while/gru_cell_11/unstackUnpack.gru_5/while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num§
-gru_5/while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp8gru_5_while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ê
gru_5/while/gru_cell_11/MatMulMatMul6gru_5/while/TensorArrayV2Read/TensorListGetItem:item:05gru_5/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
gru_5/while/gru_cell_11/BiasAddBiasAdd(gru_5/while/gru_cell_11/MatMul:product:0(gru_5/while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
'gru_5/while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_5/while/gru_cell_11/splitSplit0gru_5/while/gru_cell_11/split/split_dim:output:0(gru_5/while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split«
/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp:gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0±
 gru_5/while/gru_cell_11/MatMul_1MatMulgru_5_while_placeholder_27gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬µ
!gru_5/while/gru_cell_11/BiasAdd_1BiasAdd*gru_5/while/gru_cell_11/MatMul_1:product:0(gru_5/while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
gru_5/while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿt
)gru_5/while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_5/while/gru_cell_11/split_1SplitV*gru_5/while/gru_cell_11/BiasAdd_1:output:0&gru_5/while/gru_cell_11/Const:output:02gru_5/while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¨
gru_5/while/gru_cell_11/addAddV2&gru_5/while/gru_cell_11/split:output:0(gru_5/while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_5/while/gru_cell_11/SigmoidSigmoidgru_5/while/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
gru_5/while/gru_cell_11/add_1AddV2&gru_5/while/gru_cell_11/split:output:1(gru_5/while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_5/while/gru_cell_11/Sigmoid_1Sigmoid!gru_5/while/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
gru_5/while/gru_cell_11/mulMul%gru_5/while/gru_cell_11/Sigmoid_1:y:0(gru_5/while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¡
gru_5/while/gru_cell_11/add_2AddV2&gru_5/while/gru_cell_11/split:output:2gru_5/while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_5/while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ? 
gru_5/while/gru_cell_11/mul_1Mul%gru_5/while/gru_cell_11/beta:output:0!gru_5/while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_5/while/gru_cell_11/Sigmoid_2Sigmoid!gru_5/while/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_5/while/gru_cell_11/mul_2Mul!gru_5/while/gru_cell_11/add_2:z:0%gru_5/while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_5/while/gru_cell_11/IdentityIdentity!gru_5/while/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdë
!gru_5/while/gru_cell_11/IdentityN	IdentityN!gru_5/while/gru_cell_11/mul_2:z:0!gru_5/while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139946*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_5/while/gru_cell_11/mul_3Mul#gru_5/while/gru_cell_11/Sigmoid:y:0gru_5_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
gru_5/while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_5/while/gru_cell_11/subSub&gru_5/while/gru_cell_11/sub/x:output:0#gru_5/while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd£
gru_5/while/gru_cell_11/mul_4Mulgru_5/while/gru_cell_11/sub:z:0*gru_5/while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/while/gru_cell_11/add_3AddV2!gru_5/while/gru_cell_11/mul_3:z:0!gru_5/while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÜ
0gru_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_5_while_placeholder_1gru_5_while_placeholder!gru_5/while/gru_cell_11/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_5/while/addAddV2gru_5_while_placeholdergru_5/while/add/y:output:0*
T0*
_output_shapes
: U
gru_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_5/while/add_1AddV2$gru_5_while_gru_5_while_loop_countergru_5/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_5/while/IdentityIdentitygru_5/while/add_1:z:0^gru_5/while/NoOp*
T0*
_output_shapes
: 
gru_5/while/Identity_1Identity*gru_5_while_gru_5_while_maximum_iterations^gru_5/while/NoOp*
T0*
_output_shapes
: k
gru_5/while/Identity_2Identitygru_5/while/add:z:0^gru_5/while/NoOp*
T0*
_output_shapes
: «
gru_5/while/Identity_3Identity@gru_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_5/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_5/while/Identity_4Identity!gru_5/while/gru_cell_11/add_3:z:0^gru_5/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÝ
gru_5/while/NoOpNoOp.^gru_5/while/gru_cell_11/MatMul/ReadVariableOp0^gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp'^gru_5/while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_5_while_gru_5_strided_slice_1#gru_5_while_gru_5_strided_slice_1_0"v
8gru_5_while_gru_cell_11_matmul_1_readvariableop_resource:gru_5_while_gru_cell_11_matmul_1_readvariableop_resource_0"r
6gru_5_while_gru_cell_11_matmul_readvariableop_resource8gru_5_while_gru_cell_11_matmul_readvariableop_resource_0"d
/gru_5_while_gru_cell_11_readvariableop_resource1gru_5_while_gru_cell_11_readvariableop_resource_0"5
gru_5_while_identitygru_5/while/Identity:output:0"9
gru_5_while_identity_1gru_5/while/Identity_1:output:0"9
gru_5_while_identity_2gru_5/while/Identity_2:output:0"9
gru_5_while_identity_3gru_5/while/Identity_3:output:0"9
gru_5_while_identity_4gru_5/while/Identity_4:output:0"À
]gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_gru_5_while_tensorarrayv2read_tensorlistgetitem_gru_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2^
-gru_5/while/gru_cell_11/MatMul/ReadVariableOp-gru_5/while/gru_cell_11/MatMul/ReadVariableOp2b
/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp/gru_5/while/gru_cell_11/MatMul_1/ReadVariableOp2P
&gru_5/while/gru_cell_11/ReadVariableOp&gru_5/while/gru_cell_11/ReadVariableOp: 
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
ö
®
while_body_137154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_9_137176_0:	¬,
while_gru_cell_9_137178_0:	¬,
while_gru_cell_9_137180_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_9_137176:	¬*
while_gru_cell_9_137178:	¬*
while_gru_cell_9_137180:	d¬¢(while/gru_cell_9/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ÿ
(while/gru_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9_137176_0while_gru_cell_9_137178_0while_gru_cell_9_137180_0*
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
GPU 2J 8 *O
fJRH
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137141Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity1while/gru_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_9_137176while_gru_cell_9_137176_0"4
while_gru_cell_9_137178while_gru_cell_9_137178_0"4
while_gru_cell_9_137180while_gru_cell_9_137180_0")
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
(while/gru_cell_9/StatefulPartitionedCall(while/gru_cell_9/StatefulPartitionedCall: 
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
ÊQ

A__inference_gru_3_layer_call_and_return_conditional_losses_138293

inputs5
"gru_cell_9_readvariableop_resource:	¬<
)gru_cell_9_matmul_readvariableop_resource:	¬>
+gru_cell_9_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_9/MatMul/ReadVariableOp¢"gru_cell_9/MatMul_1/ReadVariableOp¢gru_cell_9/ReadVariableOp¢while;
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
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_9/mul_1Mulgru_cell_9/beta:output:0gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_2Sigmoidgru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_9/mul_2Mulgru_cell_9/add_2:z:0gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/IdentityIdentitygru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
gru_cell_9/IdentityN	IdentityNgru_cell_9/mul_2:z:0gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138181*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_9/mul_3Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_9/mul_4Mulgru_cell_9/sub:z:0gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_9/add_3AddV2gru_cell_9/mul_3:z:0gru_cell_9/mul_4:z:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
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
bodyR
while_body_138197*
condR
while_cond_138196*8
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
NoOpNoOp!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


#__inference_internal_grad_fn_143413
result_grads_0
result_grads_1
mul_while_gru_cell_10_beta
mul_while_gru_cell_10_add_2
identity
mulMulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_10_betamul_while_gru_cell_10_add_2*
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
Ü

$sequential_6_gru_4_while_cond_136798B
>sequential_6_gru_4_while_sequential_6_gru_4_while_loop_counterH
Dsequential_6_gru_4_while_sequential_6_gru_4_while_maximum_iterations(
$sequential_6_gru_4_while_placeholder*
&sequential_6_gru_4_while_placeholder_1*
&sequential_6_gru_4_while_placeholder_2D
@sequential_6_gru_4_while_less_sequential_6_gru_4_strided_slice_1Z
Vsequential_6_gru_4_while_sequential_6_gru_4_while_cond_136798___redundant_placeholder0Z
Vsequential_6_gru_4_while_sequential_6_gru_4_while_cond_136798___redundant_placeholder1Z
Vsequential_6_gru_4_while_sequential_6_gru_4_while_cond_136798___redundant_placeholder2Z
Vsequential_6_gru_4_while_sequential_6_gru_4_while_cond_136798___redundant_placeholder3%
!sequential_6_gru_4_while_identity
®
sequential_6/gru_4/while/LessLess$sequential_6_gru_4_while_placeholder@sequential_6_gru_4_while_less_sequential_6_gru_4_strided_slice_1*
T0*
_output_shapes
: q
!sequential_6/gru_4/while/IdentityIdentity!sequential_6/gru_4/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_6_gru_4_while_identity*sequential_6/gru_4/while/Identity:output:0*(
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
Ú
ª
while_cond_139162
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_139162___redundant_placeholder04
0while_while_cond_139162___redundant_placeholder14
0while_while_cond_139162___redundant_placeholder24
0while_while_cond_139162___redundant_placeholder3
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
B
ÿ
while_body_138197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_9_readvariableop_resource_0:	¬D
1while_gru_cell_9_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_9_readvariableop_resource:	¬B
/while_gru_cell_9_matmul_readvariableop_resource:	¬D
1while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_9/MatMul/ReadVariableOp¢(while/gru_cell_9/MatMul_1/ReadVariableOp¢while/gru_cell_9/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0while/gru_cell_9/Const:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/beta:output:0while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_2Sigmoidwhile/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/add_2:z:0while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/IdentityIdentitywhile/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
while/gru_cell_9/IdentityN	IdentityNwhile/gru_cell_9/mul_2:z:0while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138247*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_3Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_4Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_3:z:0while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
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
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 
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
B
ÿ
while_body_140975
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_9_readvariableop_resource_0:	¬D
1while_gru_cell_9_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_9_readvariableop_resource:	¬B
/while_gru_cell_9_matmul_readvariableop_resource:	¬D
1while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_9/MatMul/ReadVariableOp¢(while/gru_cell_9/MatMul_1/ReadVariableOp¢while/gru_cell_9/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0while/gru_cell_9/Const:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/beta:output:0while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_2Sigmoidwhile/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/add_2:z:0while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/IdentityIdentitywhile/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
while/gru_cell_9/IdentityN	IdentityNwhile/gru_cell_9/mul_2:z:0while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141025*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_3Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_4Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_3:z:0while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
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
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 
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
©
¹
&__inference_gru_4_layer_call_fn_141260
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallò
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_137759|
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
Ú
ª
while_cond_137857
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_137857___redundant_placeholder04
0while_while_cond_137857___redundant_placeholder14
0while_while_cond_137857___redundant_placeholder24
0while_while_cond_137857___redundant_placeholder3
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
½


H__inference_sequential_6_layer_call_and_return_conditional_losses_139998

inputs;
(gru_3_gru_cell_9_readvariableop_resource:	¬B
/gru_3_gru_cell_9_matmul_readvariableop_resource:	¬D
1gru_3_gru_cell_9_matmul_1_readvariableop_resource:	d¬<
)gru_4_gru_cell_10_readvariableop_resource:	¬C
0gru_4_gru_cell_10_matmul_readvariableop_resource:	d¬E
2gru_4_gru_cell_10_matmul_1_readvariableop_resource:	d¬<
)gru_5_gru_cell_11_readvariableop_resource:	¬C
0gru_5_gru_cell_11_matmul_readvariableop_resource:	d¬E
2gru_5_gru_cell_11_matmul_1_readvariableop_resource:	d¬8
&dense_6_matmul_readvariableop_resource:d5
'dense_6_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢&gru_3/gru_cell_9/MatMul/ReadVariableOp¢(gru_3/gru_cell_9/MatMul_1/ReadVariableOp¢gru_3/gru_cell_9/ReadVariableOp¢gru_3/while¢'gru_4/gru_cell_10/MatMul/ReadVariableOp¢)gru_4/gru_cell_10/MatMul_1/ReadVariableOp¢ gru_4/gru_cell_10/ReadVariableOp¢gru_4/while¢'gru_5/gru_cell_11/MatMul/ReadVariableOp¢)gru_5/gru_cell_11/MatMul_1/ReadVariableOp¢ gru_5/gru_cell_11/ReadVariableOp¢gru_5/whileA
gru_3/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_3/strided_sliceStridedSlicegru_3/Shape:output:0"gru_3/strided_slice/stack:output:0$gru_3/strided_slice/stack_1:output:0$gru_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_3/zeros/packedPackgru_3/strided_slice:output:0gru_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_3/zerosFillgru_3/zeros/packed:output:0gru_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_3/transpose	Transposeinputsgru_3/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿP
gru_3/Shape_1Shapegru_3/transpose:y:0*
T0*
_output_shapes
:e
gru_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_3/strided_slice_1StridedSlicegru_3/Shape_1:output:0$gru_3/strided_slice_1/stack:output:0&gru_3/strided_slice_1/stack_1:output:0&gru_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_3/TensorArrayV2TensorListReserve*gru_3/TensorArrayV2/element_shape:output:0gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_3/transpose:y:0Dgru_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_3/strided_slice_2StridedSlicegru_3/transpose:y:0$gru_3/strided_slice_2/stack:output:0&gru_3/strided_slice_2/stack_1:output:0&gru_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_3/gru_cell_9/ReadVariableOpReadVariableOp(gru_3_gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_3/gru_cell_9/unstackUnpack'gru_3/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&gru_3/gru_cell_9/MatMul/ReadVariableOpReadVariableOp/gru_3_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0¤
gru_3/gru_cell_9/MatMulMatMulgru_3/strided_slice_2:output:0.gru_3/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_3/gru_cell_9/BiasAddBiasAdd!gru_3/gru_cell_9/MatMul:product:0!gru_3/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 gru_3/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
gru_3/gru_cell_9/splitSplit)gru_3/gru_cell_9/split/split_dim:output:0!gru_3/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(gru_3/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp1gru_3_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_3/gru_cell_9/MatMul_1MatMulgru_3/zeros:output:00gru_3/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
gru_3/gru_cell_9/BiasAdd_1BiasAdd#gru_3/gru_cell_9/MatMul_1:product:0!gru_3/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
gru_3/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"gru_3/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_3/gru_cell_9/split_1SplitV#gru_3/gru_cell_9/BiasAdd_1:output:0gru_3/gru_cell_9/Const:output:0+gru_3/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_3/gru_cell_9/addAddV2gru_3/gru_cell_9/split:output:0!gru_3/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
gru_3/gru_cell_9/SigmoidSigmoidgru_3/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/add_1AddV2gru_3/gru_cell_9/split:output:1!gru_3/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_3/gru_cell_9/Sigmoid_1Sigmoidgru_3/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mulMulgru_3/gru_cell_9/Sigmoid_1:y:0!gru_3/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/add_2AddV2gru_3/gru_cell_9/split:output:2gru_3/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
gru_3/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/gru_cell_9/mul_1Mulgru_3/gru_cell_9/beta:output:0gru_3/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_3/gru_cell_9/Sigmoid_2Sigmoidgru_3/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mul_2Mulgru_3/gru_cell_9/add_2:z:0gru_3/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
gru_3/gru_cell_9/IdentityIdentitygru_3/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
gru_3/gru_cell_9/IdentityN	IdentityNgru_3/gru_cell_9/mul_2:z:0gru_3/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139554*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mul_3Mulgru_3/gru_cell_9/Sigmoid:y:0gru_3/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_3/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/gru_cell_9/subSubgru_3/gru_cell_9/sub/x:output:0gru_3/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/mul_4Mulgru_3/gru_cell_9/sub:z:0#gru_3/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/gru_cell_9/add_3AddV2gru_3/gru_cell_9/mul_3:z:0gru_3/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_3/TensorArrayV2_1TensorListReserve,gru_3/TensorArrayV2_1/element_shape:output:0gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_3/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_3/whileWhile!gru_3/while/loop_counter:output:0'gru_3/while/maximum_iterations:output:0gru_3/time:output:0gru_3/TensorArrayV2_1:handle:0gru_3/zeros:output:0gru_3/strided_slice_1:output:0=gru_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_3_gru_cell_9_readvariableop_resource/gru_3_gru_cell_9_matmul_readvariableop_resource1gru_3_gru_cell_9_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *#
bodyR
gru_3_while_body_139570*#
condR
gru_3_while_cond_139569*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_3/TensorArrayV2Stack/TensorListStackTensorListStackgru_3/while:output:3?gru_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_3/strided_slice_3StridedSlice1gru_3/TensorArrayV2Stack/TensorListStack:tensor:0$gru_3/strided_slice_3/stack:output:0&gru_3/strided_slice_3/stack_1:output:0&gru_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_3/transpose_1	Transpose1gru_3/TensorArrayV2Stack/TensorListStack:tensor:0gru_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_4/ShapeShapegru_3/transpose_1:y:0*
T0*
_output_shapes
:c
gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_4/strided_sliceStridedSlicegru_4/Shape:output:0"gru_4/strided_slice/stack:output:0$gru_4/strided_slice/stack_1:output:0$gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_4/zeros/packedPackgru_4/strided_slice:output:0gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_4/zerosFillgru_4/zeros/packed:output:0gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_4/transpose	Transposegru_3/transpose_1:y:0gru_4/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_4/Shape_1Shapegru_4/transpose:y:0*
T0*
_output_shapes
:e
gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_4/strided_slice_1StridedSlicegru_4/Shape_1:output:0$gru_4/strided_slice_1/stack:output:0&gru_4/strided_slice_1/stack_1:output:0&gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_4/TensorArrayV2TensorListReserve*gru_4/TensorArrayV2/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_4/transpose:y:0Dgru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_4/strided_slice_2StridedSlicegru_4/transpose:y:0$gru_4/strided_slice_2/stack:output:0&gru_4/strided_slice_2/stack_1:output:0&gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
 gru_4/gru_cell_10/ReadVariableOpReadVariableOp)gru_4_gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_4/gru_cell_10/unstackUnpack(gru_4/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'gru_4/gru_cell_10/MatMul/ReadVariableOpReadVariableOp0gru_4_gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¦
gru_4/gru_cell_10/MatMulMatMulgru_4/strided_slice_2:output:0/gru_4/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_4/gru_cell_10/BiasAddBiasAdd"gru_4/gru_cell_10/MatMul:product:0"gru_4/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!gru_4/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_4/gru_cell_10/splitSplit*gru_4/gru_cell_10/split/split_dim:output:0"gru_4/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)gru_4/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp2gru_4_gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0 
gru_4/gru_cell_10/MatMul_1MatMulgru_4/zeros:output:01gru_4/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
gru_4/gru_cell_10/BiasAdd_1BiasAdd$gru_4/gru_cell_10/MatMul_1:product:0"gru_4/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
gru_4/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#gru_4/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_4/gru_cell_10/split_1SplitV$gru_4/gru_cell_10/BiasAdd_1:output:0 gru_4/gru_cell_10/Const:output:0,gru_4/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_4/gru_cell_10/addAddV2 gru_4/gru_cell_10/split:output:0"gru_4/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_4/gru_cell_10/SigmoidSigmoidgru_4/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/add_1AddV2 gru_4/gru_cell_10/split:output:1"gru_4/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_4/gru_cell_10/Sigmoid_1Sigmoidgru_4/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mulMulgru_4/gru_cell_10/Sigmoid_1:y:0"gru_4/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/add_2AddV2 gru_4/gru_cell_10/split:output:2gru_4/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_4/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_4/gru_cell_10/mul_1Mulgru_4/gru_cell_10/beta:output:0gru_4/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_4/gru_cell_10/Sigmoid_2Sigmoidgru_4/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mul_2Mulgru_4/gru_cell_10/add_2:z:0gru_4/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_4/gru_cell_10/IdentityIdentitygru_4/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
gru_4/gru_cell_10/IdentityN	IdentityNgru_4/gru_cell_10/mul_2:z:0gru_4/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139717*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mul_3Mulgru_4/gru_cell_10/Sigmoid:y:0gru_4/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
gru_4/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_4/gru_cell_10/subSub gru_4/gru_cell_10/sub/x:output:0gru_4/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/mul_4Mulgru_4/gru_cell_10/sub:z:0$gru_4/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/gru_cell_10/add_3AddV2gru_4/gru_cell_10/mul_3:z:0gru_4/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_4/TensorArrayV2_1TensorListReserve,gru_4/TensorArrayV2_1/element_shape:output:0gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_4/whileWhile!gru_4/while/loop_counter:output:0'gru_4/while/maximum_iterations:output:0gru_4/time:output:0gru_4/TensorArrayV2_1:handle:0gru_4/zeros:output:0gru_4/strided_slice_1:output:0=gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_4_gru_cell_10_readvariableop_resource0gru_4_gru_cell_10_matmul_readvariableop_resource2gru_4_gru_cell_10_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *#
bodyR
gru_4_while_body_139733*#
condR
gru_4_while_cond_139732*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_4/TensorArrayV2Stack/TensorListStackTensorListStackgru_4/while:output:3?gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_4/strided_slice_3StridedSlice1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0$gru_4/strided_slice_3/stack:output:0&gru_4/strided_slice_3/stack_1:output:0&gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_4/transpose_1	Transpose1gru_4/TensorArrayV2Stack/TensorListStack:tensor:0gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    P
gru_5/ShapeShapegru_4/transpose_1:y:0*
T0*
_output_shapes
:c
gru_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_5/strided_sliceStridedSlicegru_5/Shape:output:0"gru_5/strided_slice/stack:output:0$gru_5/strided_slice/stack_1:output:0$gru_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
gru_5/zeros/packedPackgru_5/strided_slice:output:0gru_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_5/zerosFillgru_5/zeros/packed:output:0gru_5/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_5/transpose	Transposegru_4/transpose_1:y:0gru_5/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdP
gru_5/Shape_1Shapegru_5/transpose:y:0*
T0*
_output_shapes
:e
gru_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_5/strided_slice_1StridedSlicegru_5/Shape_1:output:0$gru_5/strided_slice_1/stack:output:0&gru_5/strided_slice_1/stack_1:output:0&gru_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_5/TensorArrayV2TensorListReserve*gru_5/TensorArrayV2/element_shape:output:0gru_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ò
-gru_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_5/transpose:y:0Dgru_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_5/strided_slice_2StridedSlicegru_5/transpose:y:0$gru_5/strided_slice_2/stack:output:0&gru_5/strided_slice_2/stack_1:output:0&gru_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask
 gru_5/gru_cell_11/ReadVariableOpReadVariableOp)gru_5_gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_5/gru_cell_11/unstackUnpack(gru_5/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'gru_5/gru_cell_11/MatMul/ReadVariableOpReadVariableOp0gru_5_gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0¦
gru_5/gru_cell_11/MatMulMatMulgru_5/strided_slice_2:output:0/gru_5/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_5/gru_cell_11/BiasAddBiasAdd"gru_5/gru_cell_11/MatMul:product:0"gru_5/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!gru_5/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_5/gru_cell_11/splitSplit*gru_5/gru_cell_11/split/split_dim:output:0"gru_5/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)gru_5/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp2gru_5_gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0 
gru_5/gru_cell_11/MatMul_1MatMulgru_5/zeros:output:01gru_5/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
gru_5/gru_cell_11/BiasAdd_1BiasAdd$gru_5/gru_cell_11/MatMul_1:product:0"gru_5/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
gru_5/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#gru_5/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_5/gru_cell_11/split_1SplitV$gru_5/gru_cell_11/BiasAdd_1:output:0 gru_5/gru_cell_11/Const:output:0,gru_5/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_5/gru_cell_11/addAddV2 gru_5/gru_cell_11/split:output:0"gru_5/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
gru_5/gru_cell_11/SigmoidSigmoidgru_5/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/add_1AddV2 gru_5/gru_cell_11/split:output:1"gru_5/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_5/gru_cell_11/Sigmoid_1Sigmoidgru_5/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mulMulgru_5/gru_cell_11/Sigmoid_1:y:0"gru_5/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/add_2AddV2 gru_5/gru_cell_11/split:output:2gru_5/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
gru_5/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_5/gru_cell_11/mul_1Mulgru_5/gru_cell_11/beta:output:0gru_5/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_5/gru_cell_11/Sigmoid_2Sigmoidgru_5/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mul_2Mulgru_5/gru_cell_11/add_2:z:0gru_5/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
gru_5/gru_cell_11/IdentityIdentitygru_5/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
gru_5/gru_cell_11/IdentityN	IdentityNgru_5/gru_cell_11/mul_2:z:0gru_5/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-139880*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mul_3Mulgru_5/gru_cell_11/Sigmoid:y:0gru_5/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
gru_5/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_5/gru_cell_11/subSub gru_5/gru_cell_11/sub/x:output:0gru_5/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/mul_4Mulgru_5/gru_cell_11/sub:z:0$gru_5/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_5/gru_cell_11/add_3AddV2gru_5/gru_cell_11/mul_3:z:0gru_5/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
#gru_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ê
gru_5/TensorArrayV2_1TensorListReserve,gru_5/TensorArrayV2_1/element_shape:output:0gru_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_5/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_5/whileWhile!gru_5/while/loop_counter:output:0'gru_5/while/maximum_iterations:output:0gru_5/time:output:0gru_5/TensorArrayV2_1:handle:0gru_5/zeros:output:0gru_5/strided_slice_1:output:0=gru_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_5_gru_cell_11_readvariableop_resource0gru_5_gru_cell_11_matmul_readvariableop_resource2gru_5_gru_cell_11_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *#
bodyR
gru_5_while_body_139896*#
condR
gru_5_while_cond_139895*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
6gru_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ô
(gru_5/TensorArrayV2Stack/TensorListStackTensorListStackgru_5/while:output:3?gru_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0n
gru_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_5/strided_slice_3StridedSlice1gru_5/TensorArrayV2Stack/TensorListStack:tensor:0$gru_5/strided_slice_3/stack:output:0&gru_5/strided_slice_3/stack_1:output:0&gru_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskk
gru_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_5/transpose_1	Transpose1gru_5/TensorArrayV2Stack/TensorListStack:tensor:0gru_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdda
gru_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_6/MatMulMatMulgru_5/strided_slice_3:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp'^gru_3/gru_cell_9/MatMul/ReadVariableOp)^gru_3/gru_cell_9/MatMul_1/ReadVariableOp ^gru_3/gru_cell_9/ReadVariableOp^gru_3/while(^gru_4/gru_cell_10/MatMul/ReadVariableOp*^gru_4/gru_cell_10/MatMul_1/ReadVariableOp!^gru_4/gru_cell_10/ReadVariableOp^gru_4/while(^gru_5/gru_cell_11/MatMul/ReadVariableOp*^gru_5/gru_cell_11/MatMul_1/ReadVariableOp!^gru_5/gru_cell_11/ReadVariableOp^gru_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2P
&gru_3/gru_cell_9/MatMul/ReadVariableOp&gru_3/gru_cell_9/MatMul/ReadVariableOp2T
(gru_3/gru_cell_9/MatMul_1/ReadVariableOp(gru_3/gru_cell_9/MatMul_1/ReadVariableOp2B
gru_3/gru_cell_9/ReadVariableOpgru_3/gru_cell_9/ReadVariableOp2
gru_3/whilegru_3/while2R
'gru_4/gru_cell_10/MatMul/ReadVariableOp'gru_4/gru_cell_10/MatMul/ReadVariableOp2V
)gru_4/gru_cell_10/MatMul_1/ReadVariableOp)gru_4/gru_cell_10/MatMul_1/ReadVariableOp2D
 gru_4/gru_cell_10/ReadVariableOp gru_4/gru_cell_10/ReadVariableOp2
gru_4/whilegru_4/while2R
'gru_5/gru_cell_11/MatMul/ReadVariableOp'gru_5/gru_cell_11/MatMul/ReadVariableOp2V
)gru_5/gru_cell_11/MatMul_1/ReadVariableOp)gru_5/gru_cell_11/MatMul_1/ReadVariableOp2D
 gru_5/gru_cell_11/ReadVariableOp gru_5/gru_cell_11/ReadVariableOp2
gru_5/whilegru_5/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Í
´
#__inference_internal_grad_fn_143143
result_grads_0
result_grads_1*
&mul_sequential_6_gru_3_gru_cell_9_beta+
'mul_sequential_6_gru_3_gru_cell_9_add_2
identity
mulMul&mul_sequential_6_gru_3_gru_cell_9_beta'mul_sequential_6_gru_3_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul&mul_sequential_6_gru_3_gru_cell_9_beta'mul_sequential_6_gru_3_gru_cell_9_add_2*
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
Ó
¶
#__inference_internal_grad_fn_143179
result_grads_0
result_grads_1+
'mul_sequential_6_gru_5_gru_cell_11_beta,
(mul_sequential_6_gru_5_gru_cell_11_add_2
identity 
mulMul'mul_sequential_6_gru_5_gru_cell_11_beta(mul_sequential_6_gru_5_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul'mul_sequential_6_gru_5_gru_cell_11_beta(mul_sequential_6_gru_5_gru_cell_11_add_2*
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
Ú
ª
while_cond_141141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_141141___redundant_placeholder04
0while_while_cond_141141___redundant_placeholder14
0while_while_cond_141141___redundant_placeholder24
0while_while_cond_141141___redundant_placeholder3
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


#__inference_internal_grad_fn_143611
result_grads_0
result_grads_1
mul_gru_5_gru_cell_11_beta
mul_gru_5_gru_cell_11_add_2
identity
mulMulmul_gru_5_gru_cell_11_betamul_gru_5_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_gru_5_gru_cell_11_betamul_gru_5_gru_cell_11_add_2*
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

x
#__inference_internal_grad_fn_144241
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
R

A__inference_gru_3_layer_call_and_return_conditional_losses_140904
inputs_05
"gru_cell_9_readvariableop_resource:	¬<
)gru_cell_9_matmul_readvariableop_resource:	¬>
+gru_cell_9_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_9/MatMul/ReadVariableOp¢"gru_cell_9/MatMul_1/ReadVariableOp¢gru_cell_9/ReadVariableOp¢while=
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
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_9/mul_1Mulgru_cell_9/beta:output:0gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_2Sigmoidgru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_9/mul_2Mulgru_cell_9/add_2:z:0gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/IdentityIdentitygru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
gru_cell_9/IdentityN	IdentityNgru_cell_9/mul_2:z:0gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140792*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_9/mul_3Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_9/mul_4Mulgru_cell_9/sub:z:0gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_9/add_3AddV2gru_cell_9/mul_3:z:0gru_cell_9/mul_4:z:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
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
bodyR
while_body_140808*
condR
while_cond_140807*8
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
NoOpNoOp!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
²

Ú
+__inference_gru_cell_9_layer_call_fn_142709

inputs
states_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *O
fJRH
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137291o
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
Ù

#__inference_internal_grad_fn_143719
result_grads_0
result_grads_1
mul_gru_cell_9_beta
mul_gru_cell_9_add_2
identityx
mulMulmul_gru_cell_9_betamul_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_9_betamul_gru_cell_9_add_2*
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
¹
Ò
H__inference_sequential_6_layer_call_and_return_conditional_losses_139327

inputs
gru_3_139300:	¬
gru_3_139302:	¬
gru_3_139304:	d¬
gru_4_139307:	¬
gru_4_139309:	d¬
gru_4_139311:	d¬
gru_5_139314:	¬
gru_5_139316:	d¬
gru_5_139318:	d¬ 
dense_6_139321:d
dense_6_139323:
identity¢dense_6/StatefulPartitionedCall¢gru_3/StatefulPartitionedCall¢gru_4/StatefulPartitionedCall¢gru_5/StatefulPartitionedCallø
gru_3/StatefulPartitionedCallStatefulPartitionedCallinputsgru_3_139300gru_3_139302gru_3_139304*
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_139259
gru_4/StatefulPartitionedCallStatefulPartitionedCall&gru_3/StatefulPartitionedCall:output:0gru_4_139307gru_4_139309gru_4_139311*
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_139070
gru_5/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0gru_5_139314gru_5_139316gru_5_139318*
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138881
dense_6/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0dense_6_139321dense_6_139323*
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
GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_138659w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_6/StatefulPartitionedCall^gru_3/StatefulPartitionedCall^gru_4/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2>
gru_3/StatefulPartitionedCallgru_3/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ú
ª
while_cond_140640
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_140640___redundant_placeholder04
0while_while_cond_140640___redundant_placeholder14
0while_while_cond_140640___redundant_placeholder24
0while_while_cond_140640___redundant_placeholder3
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
Ù

#__inference_internal_grad_fn_143251
result_grads_0
result_grads_1
mul_gru_cell_9_beta
mul_gru_cell_9_add_2
identityx
mulMulmul_gru_cell_9_betamul_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_9_betamul_gru_cell_9_add_2*
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
ß

#__inference_internal_grad_fn_143287
result_grads_0
result_grads_1
mul_gru_cell_10_beta
mul_gru_cell_10_add_2
identityz
mulMulmul_gru_cell_10_betamul_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_10_betamul_gru_cell_10_add_2*
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
4

A__inference_gru_4_layer_call_and_return_conditional_losses_137570

inputs%
gru_cell_10_137494:	¬%
gru_cell_10_137496:	d¬%
gru_cell_10_137498:	d¬
identity¢#gru_cell_10/StatefulPartitionedCall¢while;
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
shrink_axis_maskÉ
#gru_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_10_137494gru_cell_10_137496gru_cell_10_137498*
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137493n
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
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_10_137494gru_cell_10_137496gru_cell_10_137498*
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
bodyR
while_body_137506*
condR
while_cond_137505*8
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdt
NoOpNoOp$^gru_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2J
#gru_cell_10/StatefulPartitionedCall#gru_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ú
ª
while_cond_138196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_138196___redundant_placeholder04
0while_while_cond_138196___redundant_placeholder14
0while_while_cond_138196___redundant_placeholder24
0while_while_cond_138196___redundant_placeholder3
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
²

Ú
+__inference_gru_cell_9_layer_call_fn_142695

inputs
states_0
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *O
fJRH
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137141o
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
Ú
ª
while_cond_140807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_140807___redundant_placeholder04
0while_while_cond_140807___redundant_placeholder14
0while_while_cond_140807___redundant_placeholder24
0while_while_cond_140807___redundant_placeholder3
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
¨R

A__inference_gru_5_layer_call_and_return_conditional_losses_138881

inputs6
#gru_cell_11_readvariableop_resource:	¬=
*gru_cell_11_matmul_readvariableop_resource:	d¬?
,gru_cell_11_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_11/MatMul/ReadVariableOp¢#gru_cell_11/MatMul_1/ReadVariableOp¢gru_cell_11/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_11/ReadVariableOpReadVariableOp#gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_11/unstackUnpack"gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_11/MatMul/ReadVariableOpReadVariableOp*gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMulMatMulstrided_slice_2:output:0)gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAddBiasAddgru_cell_11/MatMul:product:0gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_11/splitSplit$gru_cell_11/split/split_dim:output:0gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_11/MatMul_1MatMulzeros:output:0+gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_11/BiasAdd_1BiasAddgru_cell_11/MatMul_1:product:0gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_11/split_1SplitVgru_cell_11/BiasAdd_1:output:0gru_cell_11/Const:output:0&gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_11/addAddV2gru_cell_11/split:output:0gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_11/SigmoidSigmoidgru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/add_1AddV2gru_cell_11/split:output:1gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_1Sigmoidgru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mulMulgru_cell_11/Sigmoid_1:y:0gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_11/add_2AddV2gru_cell_11/split:output:2gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_11/mul_1Mulgru_cell_11/beta:output:0gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/Sigmoid_2Sigmoidgru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_11/mul_2Mulgru_cell_11/add_2:z:0gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_11/IdentityIdentitygru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_11/IdentityN	IdentityNgru_cell_11/mul_2:z:0gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138769*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_11/mul_3Mulgru_cell_11/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_11/subSubgru_cell_11/sub/x:output:0gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_11/mul_4Mulgru_cell_11/sub:z:0gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_11/add_3AddV2gru_cell_11/mul_3:z:0gru_cell_11/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_11_readvariableop_resource*gru_cell_11_matmul_readvariableop_resource,gru_cell_11_matmul_1_readvariableop_resource*
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
bodyR
while_body_138785*
condR
while_cond_138784*8
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
:ÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_11/MatMul/ReadVariableOp$^gru_cell_11/MatMul_1/ReadVariableOp^gru_cell_11/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_11/MatMul/ReadVariableOp!gru_cell_11/MatMul/ReadVariableOp2J
#gru_cell_11/MatMul_1/ReadVariableOp#gru_cell_11/MatMul_1/ReadVariableOp28
gru_cell_11/ReadVariableOpgru_cell_11/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
è[
¹
$sequential_6_gru_3_while_body_136636B
>sequential_6_gru_3_while_sequential_6_gru_3_while_loop_counterH
Dsequential_6_gru_3_while_sequential_6_gru_3_while_maximum_iterations(
$sequential_6_gru_3_while_placeholder*
&sequential_6_gru_3_while_placeholder_1*
&sequential_6_gru_3_while_placeholder_2A
=sequential_6_gru_3_while_sequential_6_gru_3_strided_slice_1_0}
ysequential_6_gru_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_3_tensorarrayunstack_tensorlistfromtensor_0P
=sequential_6_gru_3_while_gru_cell_9_readvariableop_resource_0:	¬W
Dsequential_6_gru_3_while_gru_cell_9_matmul_readvariableop_resource_0:	¬Y
Fsequential_6_gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬%
!sequential_6_gru_3_while_identity'
#sequential_6_gru_3_while_identity_1'
#sequential_6_gru_3_while_identity_2'
#sequential_6_gru_3_while_identity_3'
#sequential_6_gru_3_while_identity_4?
;sequential_6_gru_3_while_sequential_6_gru_3_strided_slice_1{
wsequential_6_gru_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_3_tensorarrayunstack_tensorlistfromtensorN
;sequential_6_gru_3_while_gru_cell_9_readvariableop_resource:	¬U
Bsequential_6_gru_3_while_gru_cell_9_matmul_readvariableop_resource:	¬W
Dsequential_6_gru_3_while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢9sequential_6/gru_3/while/gru_cell_9/MatMul/ReadVariableOp¢;sequential_6/gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp¢2sequential_6/gru_3/while/gru_cell_9/ReadVariableOp
Jsequential_6/gru_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<sequential_6/gru_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_6_gru_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_3_tensorarrayunstack_tensorlistfromtensor_0$sequential_6_gru_3_while_placeholderSsequential_6/gru_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0±
2sequential_6/gru_3/while/gru_cell_9/ReadVariableOpReadVariableOp=sequential_6_gru_3_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0©
+sequential_6/gru_3/while/gru_cell_9/unstackUnpack:sequential_6/gru_3/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¿
9sequential_6/gru_3/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOpDsequential_6_gru_3_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0ï
*sequential_6/gru_3/while/gru_cell_9/MatMulMatMulCsequential_6/gru_3/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_6/gru_3/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Õ
+sequential_6/gru_3/while/gru_cell_9/BiasAddBiasAdd4sequential_6/gru_3/while/gru_cell_9/MatMul:product:04sequential_6/gru_3/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
3sequential_6/gru_3/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
)sequential_6/gru_3/while/gru_cell_9/splitSplit<sequential_6/gru_3/while/gru_cell_9/split/split_dim:output:04sequential_6/gru_3/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÃ
;sequential_6/gru_3/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOpFsequential_6_gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ö
,sequential_6/gru_3/while/gru_cell_9/MatMul_1MatMul&sequential_6_gru_3_while_placeholder_2Csequential_6/gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ù
-sequential_6/gru_3/while/gru_cell_9/BiasAdd_1BiasAdd6sequential_6/gru_3/while/gru_cell_9/MatMul_1:product:04sequential_6/gru_3/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬~
)sequential_6/gru_3/while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ
5sequential_6/gru_3/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
+sequential_6/gru_3/while/gru_cell_9/split_1SplitV6sequential_6/gru_3/while/gru_cell_9/BiasAdd_1:output:02sequential_6/gru_3/while/gru_cell_9/Const:output:0>sequential_6/gru_3/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitÌ
'sequential_6/gru_3/while/gru_cell_9/addAddV22sequential_6/gru_3/while/gru_cell_9/split:output:04sequential_6/gru_3/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+sequential_6/gru_3/while/gru_cell_9/SigmoidSigmoid+sequential_6/gru_3/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÎ
)sequential_6/gru_3/while/gru_cell_9/add_1AddV22sequential_6/gru_3/while/gru_cell_9/split:output:14sequential_6/gru_3/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_6/gru_3/while/gru_cell_9/Sigmoid_1Sigmoid-sequential_6/gru_3/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÉ
'sequential_6/gru_3/while/gru_cell_9/mulMul1sequential_6/gru_3/while/gru_cell_9/Sigmoid_1:y:04sequential_6/gru_3/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ
)sequential_6/gru_3/while/gru_cell_9/add_2AddV22sequential_6/gru_3/while/gru_cell_9/split:output:2+sequential_6/gru_3/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
(sequential_6/gru_3/while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
)sequential_6/gru_3/while/gru_cell_9/mul_1Mul1sequential_6/gru_3/while/gru_cell_9/beta:output:0-sequential_6/gru_3/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_6/gru_3/while/gru_cell_9/Sigmoid_2Sigmoid-sequential_6/gru_3/while/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
)sequential_6/gru_3/while/gru_cell_9/mul_2Mul-sequential_6/gru_3/while/gru_cell_9/add_2:z:01sequential_6/gru_3/while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_6/gru_3/while/gru_cell_9/IdentityIdentity-sequential_6/gru_3/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
-sequential_6/gru_3/while/gru_cell_9/IdentityN	IdentityN-sequential_6/gru_3/while/gru_cell_9/mul_2:z:0-sequential_6/gru_3/while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-136686*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd»
)sequential_6/gru_3/while/gru_cell_9/mul_3Mul/sequential_6/gru_3/while/gru_cell_9/Sigmoid:y:0&sequential_6_gru_3_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
)sequential_6/gru_3/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Å
'sequential_6/gru_3/while/gru_cell_9/subSub2sequential_6/gru_3/while/gru_cell_9/sub/x:output:0/sequential_6/gru_3/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
)sequential_6/gru_3/while/gru_cell_9/mul_4Mul+sequential_6/gru_3/while/gru_cell_9/sub:z:06sequential_6/gru_3/while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ
)sequential_6/gru_3/while/gru_cell_9/add_3AddV2-sequential_6/gru_3/while/gru_cell_9/mul_3:z:0-sequential_6/gru_3/while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
=sequential_6/gru_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_6_gru_3_while_placeholder_1$sequential_6_gru_3_while_placeholder-sequential_6/gru_3/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_6/gru_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_6/gru_3/while/addAddV2$sequential_6_gru_3_while_placeholder'sequential_6/gru_3/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_6/gru_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_6/gru_3/while/add_1AddV2>sequential_6_gru_3_while_sequential_6_gru_3_while_loop_counter)sequential_6/gru_3/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_6/gru_3/while/IdentityIdentity"sequential_6/gru_3/while/add_1:z:0^sequential_6/gru_3/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_6/gru_3/while/Identity_1IdentityDsequential_6_gru_3_while_sequential_6_gru_3_while_maximum_iterations^sequential_6/gru_3/while/NoOp*
T0*
_output_shapes
: 
#sequential_6/gru_3/while/Identity_2Identity sequential_6/gru_3/while/add:z:0^sequential_6/gru_3/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_6/gru_3/while/Identity_3IdentityMsequential_6/gru_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_6/gru_3/while/NoOp*
T0*
_output_shapes
: :éèÒ°
#sequential_6/gru_3/while/Identity_4Identity-sequential_6/gru_3/while/gru_cell_9/add_3:z:0^sequential_6/gru_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_6/gru_3/while/NoOpNoOp:^sequential_6/gru_3/while/gru_cell_9/MatMul/ReadVariableOp<^sequential_6/gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp3^sequential_6/gru_3/while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Dsequential_6_gru_3_while_gru_cell_9_matmul_1_readvariableop_resourceFsequential_6_gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0"
Bsequential_6_gru_3_while_gru_cell_9_matmul_readvariableop_resourceDsequential_6_gru_3_while_gru_cell_9_matmul_readvariableop_resource_0"|
;sequential_6_gru_3_while_gru_cell_9_readvariableop_resource=sequential_6_gru_3_while_gru_cell_9_readvariableop_resource_0"O
!sequential_6_gru_3_while_identity*sequential_6/gru_3/while/Identity:output:0"S
#sequential_6_gru_3_while_identity_1,sequential_6/gru_3/while/Identity_1:output:0"S
#sequential_6_gru_3_while_identity_2,sequential_6/gru_3/while/Identity_2:output:0"S
#sequential_6_gru_3_while_identity_3,sequential_6/gru_3/while/Identity_3:output:0"S
#sequential_6_gru_3_while_identity_4,sequential_6/gru_3/while/Identity_4:output:0"|
;sequential_6_gru_3_while_sequential_6_gru_3_strided_slice_1=sequential_6_gru_3_while_sequential_6_gru_3_strided_slice_1_0"ô
wsequential_6_gru_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_3_tensorarrayunstack_tensorlistfromtensorysequential_6_gru_3_while_tensorarrayv2read_tensorlistgetitem_sequential_6_gru_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2v
9sequential_6/gru_3/while/gru_cell_9/MatMul/ReadVariableOp9sequential_6/gru_3/while/gru_cell_9/MatMul/ReadVariableOp2z
;sequential_6/gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp;sequential_6/gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp2h
2sequential_6/gru_3/while/gru_cell_9/ReadVariableOp2sequential_6/gru_3/while/gru_cell_9/ReadVariableOp: 
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
Ú
ª
while_cond_137694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_137694___redundant_placeholder04
0while_while_cond_137694___redundant_placeholder14
0while_while_cond_137694___redundant_placeholder24
0while_while_cond_137694___redundant_placeholder3
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
Ú
ª
while_cond_141352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_141352___redundant_placeholder04
0while_while_cond_141352___redundant_placeholder14
0while_while_cond_141352___redundant_placeholder24
0while_while_cond_141352___redundant_placeholder3
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

x
#__inference_internal_grad_fn_143701
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
÷B

while_body_141520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_10_readvariableop_resource_0:	¬E
2while_gru_cell_10_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_10_readvariableop_resource:	¬C
0while_gru_cell_10_matmul_readvariableop_resource:	d¬E
2while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_10/MatMul/ReadVariableOp¢)while/gru_cell_10/MatMul_1/ReadVariableOp¢ while/gru_cell_10/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_10/ReadVariableOpReadVariableOp+while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_10/unstackUnpack(while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_10/BiasAddBiasAdd"while/gru_cell_10/MatMul:product:0"while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_10/splitSplit*while/gru_cell_10/split/split_dim:output:0"while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_10/MatMul_1MatMulwhile_placeholder_21while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_10/BiasAdd_1BiasAdd$while/gru_cell_10/MatMul_1:product:0"while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_10/split_1SplitV$while/gru_cell_10/BiasAdd_1:output:0 while/gru_cell_10/Const:output:0,while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_10/addAddV2 while/gru_cell_10/split:output:0"while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_10/SigmoidSigmoidwhile/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_1AddV2 while/gru_cell_10/split:output:1"while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_1Sigmoidwhile/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mulMulwhile/gru_cell_10/Sigmoid_1:y:0"while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_2AddV2 while/gru_cell_10/split:output:2while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/mul_1Mulwhile/gru_cell_10/beta:output:0while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_2Sigmoidwhile/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_2Mulwhile/gru_cell_10/add_2:z:0while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/IdentityIdentitywhile/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_10/IdentityN	IdentityNwhile/gru_cell_10/mul_2:z:0while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141570*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_3Mulwhile/gru_cell_10/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/subSub while/gru_cell_10/sub/x:output:0while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_4Mulwhile/gru_cell_10/sub:z:0$while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_3AddV2while/gru_cell_10/mul_3:z:0while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_10/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_10/MatMul/ReadVariableOp*^while/gru_cell_10/MatMul_1/ReadVariableOp!^while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_10_matmul_1_readvariableop_resource4while_gru_cell_10_matmul_1_readvariableop_resource_0"f
0while_gru_cell_10_matmul_readvariableop_resource2while_gru_cell_10_matmul_readvariableop_resource_0"X
)while_gru_cell_10_readvariableop_resource+while_gru_cell_10_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_10/MatMul/ReadVariableOp'while/gru_cell_10/MatMul/ReadVariableOp2V
)while/gru_cell_10/MatMul_1/ReadVariableOp)while/gru_cell_10/MatMul_1/ReadVariableOp2D
 while/gru_cell_10/ReadVariableOp while/gru_cell_10/ReadVariableOp: 
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
¡J
³	
gru_3_while_body_140069(
$gru_3_while_gru_3_while_loop_counter.
*gru_3_while_gru_3_while_maximum_iterations
gru_3_while_placeholder
gru_3_while_placeholder_1
gru_3_while_placeholder_2'
#gru_3_while_gru_3_strided_slice_1_0c
_gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_0C
0gru_3_while_gru_cell_9_readvariableop_resource_0:	¬J
7gru_3_while_gru_cell_9_matmul_readvariableop_resource_0:	¬L
9gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
gru_3_while_identity
gru_3_while_identity_1
gru_3_while_identity_2
gru_3_while_identity_3
gru_3_while_identity_4%
!gru_3_while_gru_3_strided_slice_1a
]gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensorA
.gru_3_while_gru_cell_9_readvariableop_resource:	¬H
5gru_3_while_gru_cell_9_matmul_readvariableop_resource:	¬J
7gru_3_while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢,gru_3/while/gru_cell_9/MatMul/ReadVariableOp¢.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp¢%gru_3/while/gru_cell_9/ReadVariableOp
=gru_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/gru_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_0gru_3_while_placeholderFgru_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
%gru_3/while/gru_cell_9/ReadVariableOpReadVariableOp0gru_3_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_3/while/gru_cell_9/unstackUnpack-gru_3/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num¥
,gru_3/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7gru_3_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0È
gru_3/while/gru_cell_9/MatMulMatMul6gru_3/while/TensorArrayV2Read/TensorListGetItem:item:04gru_3/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬®
gru_3/while/gru_cell_9/BiasAddBiasAdd'gru_3/while/gru_cell_9/MatMul:product:0'gru_3/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
&gru_3/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
gru_3/while/gru_cell_9/splitSplit/gru_3/while/gru_cell_9/split/split_dim:output:0'gru_3/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split©
.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¯
gru_3/while/gru_cell_9/MatMul_1MatMulgru_3_while_placeholder_26gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬²
 gru_3/while/gru_cell_9/BiasAdd_1BiasAdd)gru_3/while/gru_cell_9/MatMul_1:product:0'gru_3/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
gru_3/while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿs
(gru_3/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
gru_3/while/gru_cell_9/split_1SplitV)gru_3/while/gru_cell_9/BiasAdd_1:output:0%gru_3/while/gru_cell_9/Const:output:01gru_3/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¥
gru_3/while/gru_cell_9/addAddV2%gru_3/while/gru_cell_9/split:output:0'gru_3/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd{
gru_3/while/gru_cell_9/SigmoidSigmoidgru_3/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
gru_3/while/gru_cell_9/add_1AddV2%gru_3/while/gru_cell_9/split:output:1'gru_3/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_3/while/gru_cell_9/Sigmoid_1Sigmoid gru_3/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
gru_3/while/gru_cell_9/mulMul$gru_3/while/gru_cell_9/Sigmoid_1:y:0'gru_3/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/add_2AddV2%gru_3/while/gru_cell_9/split:output:2gru_3/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
gru_3/while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/while/gru_cell_9/mul_1Mul$gru_3/while/gru_cell_9/beta:output:0 gru_3/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_3/while/gru_cell_9/Sigmoid_2Sigmoid gru_3/while/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/mul_2Mul gru_3/while/gru_cell_9/add_2:z:0$gru_3/while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/IdentityIdentity gru_3/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdè
 gru_3/while/gru_cell_9/IdentityN	IdentityN gru_3/while/gru_cell_9/mul_2:z:0 gru_3/while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140119*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/mul_3Mul"gru_3/while/gru_cell_9/Sigmoid:y:0gru_3_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_3/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_3/while/gru_cell_9/subSub%gru_3/while/gru_cell_9/sub/x:output:0"gru_3/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_3/while/gru_cell_9/mul_4Mulgru_3/while/gru_cell_9/sub:z:0)gru_3/while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_3/while/gru_cell_9/add_3AddV2 gru_3/while/gru_cell_9/mul_3:z:0 gru_3/while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
0gru_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_3_while_placeholder_1gru_3_while_placeholder gru_3/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_3/while/addAddV2gru_3_while_placeholdergru_3/while/add/y:output:0*
T0*
_output_shapes
: U
gru_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_3/while/add_1AddV2$gru_3_while_gru_3_while_loop_countergru_3/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_3/while/IdentityIdentitygru_3/while/add_1:z:0^gru_3/while/NoOp*
T0*
_output_shapes
: 
gru_3/while/Identity_1Identity*gru_3_while_gru_3_while_maximum_iterations^gru_3/while/NoOp*
T0*
_output_shapes
: k
gru_3/while/Identity_2Identitygru_3/while/add:z:0^gru_3/while/NoOp*
T0*
_output_shapes
: «
gru_3/while/Identity_3Identity@gru_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_3/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_3/while/Identity_4Identity gru_3/while/gru_cell_9/add_3:z:0^gru_3/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÚ
gru_3/while/NoOpNoOp-^gru_3/while/gru_cell_9/MatMul/ReadVariableOp/^gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_3/while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_3_while_gru_3_strided_slice_1#gru_3_while_gru_3_strided_slice_1_0"t
7gru_3_while_gru_cell_9_matmul_1_readvariableop_resource9gru_3_while_gru_cell_9_matmul_1_readvariableop_resource_0"p
5gru_3_while_gru_cell_9_matmul_readvariableop_resource7gru_3_while_gru_cell_9_matmul_readvariableop_resource_0"b
.gru_3_while_gru_cell_9_readvariableop_resource0gru_3_while_gru_cell_9_readvariableop_resource_0"5
gru_3_while_identitygru_3/while/Identity:output:0"9
gru_3_while_identity_1gru_3/while/Identity_1:output:0"9
gru_3_while_identity_2gru_3/while/Identity_2:output:0"9
gru_3_while_identity_3gru_3/while/Identity_3:output:0"9
gru_3_while_identity_4gru_3/while/Identity_4:output:0"À
]gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_gru_3_while_tensorarrayv2read_tensorlistgetitem_gru_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2\
,gru_3/while/gru_cell_9/MatMul/ReadVariableOp,gru_3/while/gru_cell_9/MatMul/ReadVariableOp2`
.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp.gru_3/while/gru_cell_9/MatMul_1/ReadVariableOp2N
%gru_3/while/gru_cell_9/ReadVariableOp%gru_3/while/gru_cell_9/ReadVariableOp: 
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
þ

#__inference_internal_grad_fn_143269
result_grads_0
result_grads_1
mul_while_gru_cell_9_beta
mul_while_gru_cell_9_add_2
identity
mulMulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_while_gru_cell_9_betamul_while_gru_cell_9_add_2*
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
Ú
ª
while_cond_138544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_138544___redundant_placeholder04
0while_while_cond_138544___redundant_placeholder14
0while_while_cond_138544___redundant_placeholder24
0while_while_cond_138544___redundant_placeholder3
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
ÿ
·
&__inference_gru_3_layer_call_fn_140559

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallç
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_138293s
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
áR

A__inference_gru_4_layer_call_and_return_conditional_losses_141616
inputs_06
#gru_cell_10_readvariableop_resource:	¬=
*gru_cell_10_matmul_readvariableop_resource:	d¬?
,gru_cell_10_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_10/MatMul/ReadVariableOp¢#gru_cell_10/MatMul_1/ReadVariableOp¢gru_cell_10/ReadVariableOp¢while=
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
shrink_axis_mask
gru_cell_10/ReadVariableOpReadVariableOp#gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_10/unstackUnpack"gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_10/MatMul/ReadVariableOpReadVariableOp*gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMulMatMulstrided_slice_2:output:0)gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAddBiasAddgru_cell_10/MatMul:product:0gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_10/splitSplit$gru_cell_10/split/split_dim:output:0gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMul_1MatMulzeros:output:0+gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAdd_1BiasAddgru_cell_10/MatMul_1:product:0gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_10/split_1SplitVgru_cell_10/BiasAdd_1:output:0gru_cell_10/Const:output:0&gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_10/addAddV2gru_cell_10/split:output:0gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_10/SigmoidSigmoidgru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/add_1AddV2gru_cell_10/split:output:1gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_1Sigmoidgru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mulMulgru_cell_10/Sigmoid_1:y:0gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_10/add_2AddV2gru_cell_10/split:output:2gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_10/mul_1Mulgru_cell_10/beta:output:0gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_2Sigmoidgru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_10/mul_2Mulgru_cell_10/add_2:z:0gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/IdentityIdentitygru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_10/IdentityN	IdentityNgru_cell_10/mul_2:z:0gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141504*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_10/mul_3Mulgru_cell_10/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_10/subSubgru_cell_10/sub/x:output:0gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mul_4Mulgru_cell_10/sub:z:0gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_10/add_3AddV2gru_cell_10/mul_3:z:0gru_cell_10/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_10_readvariableop_resource*gru_cell_10_matmul_readvariableop_resource,gru_cell_10_matmul_1_readvariableop_resource*
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
bodyR
while_body_141520*
condR
while_cond_141519*8
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿdµ
NoOpNoOp"^gru_cell_10/MatMul/ReadVariableOp$^gru_cell_10/MatMul_1/ReadVariableOp^gru_cell_10/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd: : : 2F
!gru_cell_10/MatMul/ReadVariableOp!gru_cell_10/MatMul/ReadVariableOp2J
#gru_cell_10/MatMul_1/ReadVariableOp#gru_cell_10/MatMul_1/ReadVariableOp28
gru_cell_10/ReadVariableOpgru_cell_10/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0
´

Û
,__inference_gru_cell_11_layer_call_fn_142949

inputs
states_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity

identity_1¢StatefulPartitionedCall
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
GPU 2J 8 *P
fKRI
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137995o
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
Æ

¤
$__inference_signature_wrapper_140526
gru_3_input
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
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallgru_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
GPU 2J 8 **
f%R#
!__inference__wrapped_model_137064o
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
_user_specified_namegru_3_input
Ú
ª
while_cond_137153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_137153___redundant_placeholder04
0while_while_cond_137153___redundant_placeholder14
0while_while_cond_137153___redundant_placeholder24
0while_while_cond_137153___redundant_placeholder3
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


#__inference_internal_grad_fn_143485
result_grads_0
result_grads_1
mul_gru_4_gru_cell_10_beta
mul_gru_4_gru_cell_10_add_2
identity
mulMulmul_gru_4_gru_cell_10_betamul_gru_4_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_gru_4_gru_cell_10_betamul_gru_4_gru_cell_10_add_2*
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


#__inference_internal_grad_fn_143593
result_grads_0
result_grads_1
mul_gru_4_gru_cell_10_beta
mul_gru_4_gru_cell_10_add_2
identity
mulMulmul_gru_4_gru_cell_10_betamul_gru_4_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_gru_4_gru_cell_10_betamul_gru_4_gru_cell_10_add_2*
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
ö
®
while_body_137343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_9_137365_0:	¬,
while_gru_cell_9_137367_0:	¬,
while_gru_cell_9_137369_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_9_137365:	¬*
while_gru_cell_9_137367:	¬*
while_gru_cell_9_137369:	d¬¢(while/gru_cell_9/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0ÿ
(while/gru_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9_137365_0while_gru_cell_9_137367_0while_gru_cell_9_137369_0*
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
GPU 2J 8 *O
fJRH
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137291Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity1while/gru_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw

while/NoOpNoOp)^while/gru_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "4
while_gru_cell_9_137365while_gru_cell_9_137365_0"4
while_gru_cell_9_137367while_gru_cell_9_137367_0"4
while_gru_cell_9_137369while_gru_cell_9_137369_0")
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
(while/gru_cell_9/StatefulPartitionedCall(while/gru_cell_9/StatefulPartitionedCall: 
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
!
Û
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137995

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-137981*:
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

x
#__inference_internal_grad_fn_143863
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
©
¨
#__inference_internal_grad_fn_143647
result_grads_0
result_grads_1$
 mul_gru_4_while_gru_cell_10_beta%
!mul_gru_4_while_gru_cell_10_add_2
identity
mulMul mul_gru_4_while_gru_cell_10_beta!mul_gru_4_while_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul mul_gru_4_while_gru_cell_10_beta!mul_gru_4_while_gru_cell_10_add_2*
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
ß

#__inference_internal_grad_fn_143899
result_grads_0
result_grads_1
mul_gru_cell_10_beta
mul_gru_cell_10_add_2
identityz
mulMulmul_gru_cell_10_betamul_gru_cell_10_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
mul_1Mulmul_gru_cell_10_betamul_gru_cell_10_add_2*
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
ÿ
·
&__inference_gru_4_layer_call_fn_141282

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallç
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_139070s
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

x
#__inference_internal_grad_fn_144259
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
!
Ý
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_142995

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142981*:
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
ÿ
·
&__inference_gru_3_layer_call_fn_140570

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallç
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_139259s
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
Ú
ª
while_cond_141686
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_141686___redundant_placeholder04
0while_while_cond_141686___redundant_placeholder14
0while_while_cond_141686___redundant_placeholder24
0while_while_cond_141686___redundant_placeholder3
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
4
ÿ
A__inference_gru_3_layer_call_and_return_conditional_losses_137218

inputs$
gru_cell_9_137142:	¬$
gru_cell_9_137144:	¬$
gru_cell_9_137146:	d¬
identity¢"gru_cell_9/StatefulPartitionedCall¢while;
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
shrink_axis_maskÄ
"gru_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9_137142gru_cell_9_137144gru_cell_9_137146*
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
GPU 2J 8 *O
fJRH
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137141n
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
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9_137142gru_cell_9_137144gru_cell_9_137146*
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
bodyR
while_body_137154*
condR
while_cond_137153*8
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
NoOpNoOp#^gru_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"gru_cell_9/StatefulPartitionedCall"gru_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷B

while_body_141854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_10_readvariableop_resource_0:	¬E
2while_gru_cell_10_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_10_readvariableop_resource:	¬C
0while_gru_cell_10_matmul_readvariableop_resource:	d¬E
2while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_10/MatMul/ReadVariableOp¢)while/gru_cell_10/MatMul_1/ReadVariableOp¢ while/gru_cell_10/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_10/ReadVariableOpReadVariableOp+while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_10/unstackUnpack(while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_10/BiasAddBiasAdd"while/gru_cell_10/MatMul:product:0"while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_10/splitSplit*while/gru_cell_10/split/split_dim:output:0"while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_10/MatMul_1MatMulwhile_placeholder_21while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_10/BiasAdd_1BiasAdd$while/gru_cell_10/MatMul_1:product:0"while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_10/split_1SplitV$while/gru_cell_10/BiasAdd_1:output:0 while/gru_cell_10/Const:output:0,while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_10/addAddV2 while/gru_cell_10/split:output:0"while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_10/SigmoidSigmoidwhile/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_1AddV2 while/gru_cell_10/split:output:1"while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_1Sigmoidwhile/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mulMulwhile/gru_cell_10/Sigmoid_1:y:0"while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_2AddV2 while/gru_cell_10/split:output:2while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/mul_1Mulwhile/gru_cell_10/beta:output:0while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_2Sigmoidwhile/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_2Mulwhile/gru_cell_10/add_2:z:0while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/IdentityIdentitywhile/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_10/IdentityN	IdentityNwhile/gru_cell_10/mul_2:z:0while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141904*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_3Mulwhile/gru_cell_10/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/subSub while/gru_cell_10/sub/x:output:0while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_4Mulwhile/gru_cell_10/sub:z:0$while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_3AddV2while/gru_cell_10/mul_3:z:0while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_10/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_10/MatMul/ReadVariableOp*^while/gru_cell_10/MatMul_1/ReadVariableOp!^while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_10_matmul_1_readvariableop_resource4while_gru_cell_10_matmul_1_readvariableop_resource_0"f
0while_gru_cell_10_matmul_readvariableop_resource2while_gru_cell_10_matmul_readvariableop_resource_0"X
)while_gru_cell_10_readvariableop_resource+while_gru_cell_10_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_10/MatMul/ReadVariableOp'while/gru_cell_10/MatMul/ReadVariableOp2V
)while/gru_cell_10/MatMul_1/ReadVariableOp)while/gru_cell_10/MatMul_1/ReadVariableOp2D
 while/gru_cell_10/ReadVariableOp while/gru_cell_10/ReadVariableOp: 
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
þ

#__inference_internal_grad_fn_143575
result_grads_0
result_grads_1
mul_gru_3_gru_cell_9_beta
mul_gru_3_gru_cell_9_add_2
identity
mulMulmul_gru_3_gru_cell_9_betamul_gru_3_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
mul_1Mulmul_gru_3_gru_cell_9_betamul_gru_3_gru_cell_9_add_2*
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
ÊQ

A__inference_gru_3_layer_call_and_return_conditional_losses_141071

inputs5
"gru_cell_9_readvariableop_resource:	¬<
)gru_cell_9_matmul_readvariableop_resource:	¬>
+gru_cell_9_matmul_1_readvariableop_resource:	d¬
identity¢ gru_cell_9/MatMul/ReadVariableOp¢"gru_cell_9/MatMul_1/ReadVariableOp¢gru_cell_9/ReadVariableOp¢while;
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
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0w
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿg
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
gru_cell_9/mul_1Mulgru_cell_9/beta:output:0gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/Sigmoid_2Sigmoidgru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
gru_cell_9/mul_2Mulgru_cell_9/add_2:z:0gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
gru_cell_9/IdentityIdentitygru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
gru_cell_9/IdentityN	IdentityNgru_cell_9/mul_2:z:0gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140959*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdq
gru_cell_9/mul_3Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_9/mul_4Mulgru_cell_9/sub:z:0gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
gru_cell_9/add_3AddV2gru_cell_9/mul_3:z:0gru_cell_9/mul_4:z:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
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
bodyR
while_body_140975*
condR
while_cond_140974*8
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
NoOpNoOp!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : 2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
B
ÿ
while_body_140808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_9_readvariableop_resource_0:	¬D
1while_gru_cell_9_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_9_readvariableop_resource:	¬B
/while_gru_cell_9_matmul_readvariableop_resource:	¬D
1while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_9/MatMul/ReadVariableOp¢(while/gru_cell_9/MatMul_1/ReadVariableOp¢while/gru_cell_9/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0while/gru_cell_9/Const:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/beta:output:0while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_2Sigmoidwhile/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/add_2:z:0while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/IdentityIdentitywhile/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
while/gru_cell_9/IdentityN	IdentityNwhile/gru_cell_9/mul_2:z:0while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140858*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_3Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_4Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_3:z:0while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
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
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 
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

x
#__inference_internal_grad_fn_143881
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
È
×
H__inference_sequential_6_layer_call_and_return_conditional_losses_139409
gru_3_input
gru_3_139382:	¬
gru_3_139384:	¬
gru_3_139386:	d¬
gru_4_139389:	¬
gru_4_139391:	d¬
gru_4_139393:	d¬
gru_5_139396:	¬
gru_5_139398:	d¬
gru_5_139400:	d¬ 
dense_6_139403:d
dense_6_139405:
identity¢dense_6/StatefulPartitionedCall¢gru_3/StatefulPartitionedCall¢gru_4/StatefulPartitionedCall¢gru_5/StatefulPartitionedCallý
gru_3/StatefulPartitionedCallStatefulPartitionedCallgru_3_inputgru_3_139382gru_3_139384gru_3_139386*
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
GPU 2J 8 *J
fERC
A__inference_gru_3_layer_call_and_return_conditional_losses_138293
gru_4/StatefulPartitionedCallStatefulPartitionedCall&gru_3/StatefulPartitionedCall:output:0gru_4_139389gru_4_139391gru_4_139393*
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_138467
gru_5/StatefulPartitionedCallStatefulPartitionedCall&gru_4/StatefulPartitionedCall:output:0gru_5_139396gru_5_139398gru_5_139400*
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138641
dense_6/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0dense_6_139403dense_6_139405*
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
GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_138659w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp ^dense_6/StatefulPartitionedCall^gru_3/StatefulPartitionedCall^gru_4/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2>
gru_3/StatefulPartitionedCallgru_3/StatefulPartitionedCall2>
gru_4/StatefulPartitionedCallgru_4/StatefulPartitionedCall2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_3_input
!
Ü
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142801

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142787*:
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
B
ÿ
while_body_140641
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_9_readvariableop_resource_0:	¬D
1while_gru_cell_9_matmul_readvariableop_resource_0:	¬F
3while_gru_cell_9_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_9_readvariableop_resource:	¬B
/while_gru_cell_9_matmul_readvariableop_resource:	¬D
1while_gru_cell_9_matmul_1_readvariableop_resource:	d¬¢&while/gru_cell_9/MatMul/ReadVariableOp¢(while/gru_cell_9/MatMul_1/ReadVariableOp¢while/gru_cell_9/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0¶
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬ 
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬k
while/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿm
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0while/gru_cell_9/Const:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
while/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/beta:output:0while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/Sigmoid_2Sigmoidwhile/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/add_2:z:0while/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
while/gru_cell_9/IdentityIdentitywhile/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÖ
while/gru_cell_9/IdentityN	IdentityNwhile/gru_cell_9/mul_2:z:0while/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140691*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_3Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/mul_4Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_3:z:0while/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÂ

while/NoOpNoOp'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
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
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 
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
£
¦
#__inference_internal_grad_fn_143629
result_grads_0
result_grads_1#
mul_gru_3_while_gru_cell_9_beta$
 mul_gru_3_while_gru_cell_9_add_2
identity
mulMulmul_gru_3_while_gru_cell_9_beta mul_gru_3_while_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mulmul_gru_3_while_gru_cell_9_beta mul_gru_3_while_gru_cell_9_add_2*
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
÷B

while_body_138785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_11_readvariableop_resource_0:	¬E
2while_gru_cell_11_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_11_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_11_readvariableop_resource:	¬C
0while_gru_cell_11_matmul_readvariableop_resource:	d¬E
2while_gru_cell_11_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_11/MatMul/ReadVariableOp¢)while/gru_cell_11/MatMul_1/ReadVariableOp¢ while/gru_cell_11/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_11/ReadVariableOpReadVariableOp+while_gru_cell_11_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_11/unstackUnpack(while/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_11/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_11/BiasAddBiasAdd"while/gru_cell_11/MatMul:product:0"while/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_11/splitSplit*while/gru_cell_11/split/split_dim:output:0"while/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_11/MatMul_1MatMulwhile_placeholder_21while/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_11/BiasAdd_1BiasAdd$while/gru_cell_11/MatMul_1:product:0"while/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_11/split_1SplitV$while/gru_cell_11/BiasAdd_1:output:0 while/gru_cell_11/Const:output:0,while/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_11/addAddV2 while/gru_cell_11/split:output:0"while/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_11/SigmoidSigmoidwhile/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_1AddV2 while/gru_cell_11/split:output:1"while/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_1Sigmoidwhile/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mulMulwhile/gru_cell_11/Sigmoid_1:y:0"while/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_2AddV2 while/gru_cell_11/split:output:2while/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/mul_1Mulwhile/gru_cell_11/beta:output:0while/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/Sigmoid_2Sigmoidwhile/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_2Mulwhile/gru_cell_11/add_2:z:0while/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_11/IdentityIdentitywhile/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_11/IdentityN	IdentityNwhile/gru_cell_11/mul_2:z:0while/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138835*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_3Mulwhile/gru_cell_11/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_11/subSub while/gru_cell_11/sub/x:output:0while/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/mul_4Mulwhile/gru_cell_11/sub:z:0$while/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_11/add_3AddV2while/gru_cell_11/mul_3:z:0while/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_11/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_11/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_11/MatMul/ReadVariableOp*^while/gru_cell_11/MatMul_1/ReadVariableOp!^while/gru_cell_11/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_11_matmul_1_readvariableop_resource4while_gru_cell_11_matmul_1_readvariableop_resource_0"f
0while_gru_cell_11_matmul_readvariableop_resource2while_gru_cell_11_matmul_readvariableop_resource_0"X
)while_gru_cell_11_readvariableop_resource+while_gru_cell_11_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_11/MatMul/ReadVariableOp'while/gru_cell_11/MatMul/ReadVariableOp2V
)while/gru_cell_11/MatMul_1/ReadVariableOp)while/gru_cell_11/MatMul_1/ReadVariableOp2D
 while/gru_cell_11/ReadVariableOp while/gru_cell_11/ReadVariableOp: 
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
«Ç
©
!__inference__wrapped_model_137064
gru_3_inputH
5sequential_6_gru_3_gru_cell_9_readvariableop_resource:	¬O
<sequential_6_gru_3_gru_cell_9_matmul_readvariableop_resource:	¬Q
>sequential_6_gru_3_gru_cell_9_matmul_1_readvariableop_resource:	d¬I
6sequential_6_gru_4_gru_cell_10_readvariableop_resource:	¬P
=sequential_6_gru_4_gru_cell_10_matmul_readvariableop_resource:	d¬R
?sequential_6_gru_4_gru_cell_10_matmul_1_readvariableop_resource:	d¬I
6sequential_6_gru_5_gru_cell_11_readvariableop_resource:	¬P
=sequential_6_gru_5_gru_cell_11_matmul_readvariableop_resource:	d¬R
?sequential_6_gru_5_gru_cell_11_matmul_1_readvariableop_resource:	d¬E
3sequential_6_dense_6_matmul_readvariableop_resource:dB
4sequential_6_dense_6_biasadd_readvariableop_resource:
identity¢+sequential_6/dense_6/BiasAdd/ReadVariableOp¢*sequential_6/dense_6/MatMul/ReadVariableOp¢3sequential_6/gru_3/gru_cell_9/MatMul/ReadVariableOp¢5sequential_6/gru_3/gru_cell_9/MatMul_1/ReadVariableOp¢,sequential_6/gru_3/gru_cell_9/ReadVariableOp¢sequential_6/gru_3/while¢4sequential_6/gru_4/gru_cell_10/MatMul/ReadVariableOp¢6sequential_6/gru_4/gru_cell_10/MatMul_1/ReadVariableOp¢-sequential_6/gru_4/gru_cell_10/ReadVariableOp¢sequential_6/gru_4/while¢4sequential_6/gru_5/gru_cell_11/MatMul/ReadVariableOp¢6sequential_6/gru_5/gru_cell_11/MatMul_1/ReadVariableOp¢-sequential_6/gru_5/gru_cell_11/ReadVariableOp¢sequential_6/gru_5/whileS
sequential_6/gru_3/ShapeShapegru_3_input*
T0*
_output_shapes
:p
&sequential_6/gru_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_6/gru_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_6/gru_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_6/gru_3/strided_sliceStridedSlice!sequential_6/gru_3/Shape:output:0/sequential_6/gru_3/strided_slice/stack:output:01sequential_6/gru_3/strided_slice/stack_1:output:01sequential_6/gru_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_6/gru_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¬
sequential_6/gru_3/zeros/packedPack)sequential_6/gru_3/strided_slice:output:0*sequential_6/gru_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_6/gru_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_6/gru_3/zerosFill(sequential_6/gru_3/zeros/packed:output:0'sequential_6/gru_3/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
!sequential_6/gru_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_6/gru_3/transpose	Transposegru_3_input*sequential_6/gru_3/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿj
sequential_6/gru_3/Shape_1Shape sequential_6/gru_3/transpose:y:0*
T0*
_output_shapes
:r
(sequential_6/gru_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/gru_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_6/gru_3/strided_slice_1StridedSlice#sequential_6/gru_3/Shape_1:output:01sequential_6/gru_3/strided_slice_1/stack:output:03sequential_6/gru_3/strided_slice_1/stack_1:output:03sequential_6/gru_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_6/gru_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_6/gru_3/TensorArrayV2TensorListReserve7sequential_6/gru_3/TensorArrayV2/element_shape:output:0+sequential_6/gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_6/gru_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
:sequential_6/gru_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_6/gru_3/transpose:y:0Qsequential_6/gru_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_6/gru_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/gru_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_6/gru_3/strided_slice_2StridedSlice sequential_6/gru_3/transpose:y:01sequential_6/gru_3/strided_slice_2/stack:output:03sequential_6/gru_3/strided_slice_2/stack_1:output:03sequential_6/gru_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask£
,sequential_6/gru_3/gru_cell_9/ReadVariableOpReadVariableOp5sequential_6_gru_3_gru_cell_9_readvariableop_resource*
_output_shapes
:	¬*
dtype0
%sequential_6/gru_3/gru_cell_9/unstackUnpack4sequential_6/gru_3/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num±
3sequential_6/gru_3/gru_cell_9/MatMul/ReadVariableOpReadVariableOp<sequential_6_gru_3_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	¬*
dtype0Ë
$sequential_6/gru_3/gru_cell_9/MatMulMatMul+sequential_6/gru_3/strided_slice_2:output:0;sequential_6/gru_3/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ã
%sequential_6/gru_3/gru_cell_9/BiasAddBiasAdd.sequential_6/gru_3/gru_cell_9/MatMul:product:0.sequential_6/gru_3/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
-sequential_6/gru_3/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#sequential_6/gru_3/gru_cell_9/splitSplit6sequential_6/gru_3/gru_cell_9/split/split_dim:output:0.sequential_6/gru_3/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitµ
5sequential_6/gru_3/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp>sequential_6_gru_3_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Å
&sequential_6/gru_3/gru_cell_9/MatMul_1MatMul!sequential_6/gru_3/zeros:output:0=sequential_6/gru_3/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ç
'sequential_6/gru_3/gru_cell_9/BiasAdd_1BiasAdd0sequential_6/gru_3/gru_cell_9/MatMul_1:product:0.sequential_6/gru_3/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬x
#sequential_6/gru_3/gru_cell_9/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿz
/sequential_6/gru_3/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%sequential_6/gru_3/gru_cell_9/split_1SplitV0sequential_6/gru_3/gru_cell_9/BiasAdd_1:output:0,sequential_6/gru_3/gru_cell_9/Const:output:08sequential_6/gru_3/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitº
!sequential_6/gru_3/gru_cell_9/addAddV2,sequential_6/gru_3/gru_cell_9/split:output:0.sequential_6/gru_3/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%sequential_6/gru_3/gru_cell_9/SigmoidSigmoid%sequential_6/gru_3/gru_cell_9/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¼
#sequential_6/gru_3/gru_cell_9/add_1AddV2,sequential_6/gru_3/gru_cell_9/split:output:1.sequential_6/gru_3/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_6/gru_3/gru_cell_9/Sigmoid_1Sigmoid'sequential_6/gru_3/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd·
!sequential_6/gru_3/gru_cell_9/mulMul+sequential_6/gru_3/gru_cell_9/Sigmoid_1:y:0.sequential_6/gru_3/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
#sequential_6/gru_3/gru_cell_9/add_2AddV2,sequential_6/gru_3/gru_cell_9/split:output:2%sequential_6/gru_3/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
"sequential_6/gru_3/gru_cell_9/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
#sequential_6/gru_3/gru_cell_9/mul_1Mul+sequential_6/gru_3/gru_cell_9/beta:output:0'sequential_6/gru_3/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_6/gru_3/gru_cell_9/Sigmoid_2Sigmoid'sequential_6/gru_3/gru_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd²
#sequential_6/gru_3/gru_cell_9/mul_2Mul'sequential_6/gru_3/gru_cell_9/add_2:z:0+sequential_6/gru_3/gru_cell_9/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_6/gru_3/gru_cell_9/IdentityIdentity'sequential_6/gru_3/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdý
'sequential_6/gru_3/gru_cell_9/IdentityN	IdentityN'sequential_6/gru_3/gru_cell_9/mul_2:z:0'sequential_6/gru_3/gru_cell_9/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-136620*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿdª
#sequential_6/gru_3/gru_cell_9/mul_3Mul)sequential_6/gru_3/gru_cell_9/Sigmoid:y:0!sequential_6/gru_3/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential_6/gru_3/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential_6/gru_3/gru_cell_9/subSub,sequential_6/gru_3/gru_cell_9/sub/x:output:0)sequential_6/gru_3/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
#sequential_6/gru_3/gru_cell_9/mul_4Mul%sequential_6/gru_3/gru_cell_9/sub:z:00sequential_6/gru_3/gru_cell_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
#sequential_6/gru_3/gru_cell_9/add_3AddV2'sequential_6/gru_3/gru_cell_9/mul_3:z:0'sequential_6/gru_3/gru_cell_9/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential_6/gru_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ñ
"sequential_6/gru_3/TensorArrayV2_1TensorListReserve9sequential_6/gru_3/TensorArrayV2_1/element_shape:output:0+sequential_6/gru_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_6/gru_3/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_6/gru_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_6/gru_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ²
sequential_6/gru_3/whileWhile.sequential_6/gru_3/while/loop_counter:output:04sequential_6/gru_3/while/maximum_iterations:output:0 sequential_6/gru_3/time:output:0+sequential_6/gru_3/TensorArrayV2_1:handle:0!sequential_6/gru_3/zeros:output:0+sequential_6/gru_3/strided_slice_1:output:0Jsequential_6/gru_3/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_6_gru_3_gru_cell_9_readvariableop_resource<sequential_6_gru_3_gru_cell_9_matmul_readvariableop_resource>sequential_6_gru_3_gru_cell_9_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *0
body(R&
$sequential_6_gru_3_while_body_136636*0
cond(R&
$sequential_6_gru_3_while_cond_136635*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Csequential_6/gru_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   û
5sequential_6/gru_3/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_6/gru_3/while:output:3Lsequential_6/gru_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0{
(sequential_6/gru_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_6/gru_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_6/gru_3/strided_slice_3StridedSlice>sequential_6/gru_3/TensorArrayV2Stack/TensorListStack:tensor:01sequential_6/gru_3/strided_slice_3/stack:output:03sequential_6/gru_3/strided_slice_3/stack_1:output:03sequential_6/gru_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskx
#sequential_6/gru_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_6/gru_3/transpose_1	Transpose>sequential_6/gru_3/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_6/gru_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿddn
sequential_6/gru_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
sequential_6/gru_4/ShapeShape"sequential_6/gru_3/transpose_1:y:0*
T0*
_output_shapes
:p
&sequential_6/gru_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_6/gru_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_6/gru_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_6/gru_4/strided_sliceStridedSlice!sequential_6/gru_4/Shape:output:0/sequential_6/gru_4/strided_slice/stack:output:01sequential_6/gru_4/strided_slice/stack_1:output:01sequential_6/gru_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_6/gru_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¬
sequential_6/gru_4/zeros/packedPack)sequential_6/gru_4/strided_slice:output:0*sequential_6/gru_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_6/gru_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_6/gru_4/zerosFill(sequential_6/gru_4/zeros/packed:output:0'sequential_6/gru_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
!sequential_6/gru_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
sequential_6/gru_4/transpose	Transpose"sequential_6/gru_3/transpose_1:y:0*sequential_6/gru_4/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdj
sequential_6/gru_4/Shape_1Shape sequential_6/gru_4/transpose:y:0*
T0*
_output_shapes
:r
(sequential_6/gru_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/gru_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_6/gru_4/strided_slice_1StridedSlice#sequential_6/gru_4/Shape_1:output:01sequential_6/gru_4/strided_slice_1/stack:output:03sequential_6/gru_4/strided_slice_1/stack_1:output:03sequential_6/gru_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_6/gru_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_6/gru_4/TensorArrayV2TensorListReserve7sequential_6/gru_4/TensorArrayV2/element_shape:output:0+sequential_6/gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_6/gru_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
:sequential_6/gru_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_6/gru_4/transpose:y:0Qsequential_6/gru_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_6/gru_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/gru_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_6/gru_4/strided_slice_2StridedSlice sequential_6/gru_4/transpose:y:01sequential_6/gru_4/strided_slice_2/stack:output:03sequential_6/gru_4/strided_slice_2/stack_1:output:03sequential_6/gru_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask¥
-sequential_6/gru_4/gru_cell_10/ReadVariableOpReadVariableOp6sequential_6_gru_4_gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0
&sequential_6/gru_4/gru_cell_10/unstackUnpack5sequential_6/gru_4/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num³
4sequential_6/gru_4/gru_cell_10/MatMul/ReadVariableOpReadVariableOp=sequential_6_gru_4_gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Í
%sequential_6/gru_4/gru_cell_10/MatMulMatMul+sequential_6/gru_4/strided_slice_2:output:0<sequential_6/gru_4/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Æ
&sequential_6/gru_4/gru_cell_10/BiasAddBiasAdd/sequential_6/gru_4/gru_cell_10/MatMul:product:0/sequential_6/gru_4/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
.sequential_6/gru_4/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$sequential_6/gru_4/gru_cell_10/splitSplit7sequential_6/gru_4/gru_cell_10/split/split_dim:output:0/sequential_6/gru_4/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split·
6sequential_6/gru_4/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp?sequential_6_gru_4_gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Ç
'sequential_6/gru_4/gru_cell_10/MatMul_1MatMul!sequential_6/gru_4/zeros:output:0>sequential_6/gru_4/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ê
(sequential_6/gru_4/gru_cell_10/BiasAdd_1BiasAdd1sequential_6/gru_4/gru_cell_10/MatMul_1:product:0/sequential_6/gru_4/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
$sequential_6/gru_4/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ{
0sequential_6/gru_4/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&sequential_6/gru_4/gru_cell_10/split_1SplitV1sequential_6/gru_4/gru_cell_10/BiasAdd_1:output:0-sequential_6/gru_4/gru_cell_10/Const:output:09sequential_6/gru_4/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split½
"sequential_6/gru_4/gru_cell_10/addAddV2-sequential_6/gru_4/gru_cell_10/split:output:0/sequential_6/gru_4/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_6/gru_4/gru_cell_10/SigmoidSigmoid&sequential_6/gru_4/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¿
$sequential_6/gru_4/gru_cell_10/add_1AddV2-sequential_6/gru_4/gru_cell_10/split:output:1/sequential_6/gru_4/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_6/gru_4/gru_cell_10/Sigmoid_1Sigmoid(sequential_6/gru_4/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdº
"sequential_6/gru_4/gru_cell_10/mulMul,sequential_6/gru_4/gru_cell_10/Sigmoid_1:y:0/sequential_6/gru_4/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¶
$sequential_6/gru_4/gru_cell_10/add_2AddV2-sequential_6/gru_4/gru_cell_10/split:output:2&sequential_6/gru_4/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential_6/gru_4/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
$sequential_6/gru_4/gru_cell_10/mul_1Mul,sequential_6/gru_4/gru_cell_10/beta:output:0(sequential_6/gru_4/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_6/gru_4/gru_cell_10/Sigmoid_2Sigmoid(sequential_6/gru_4/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
$sequential_6/gru_4/gru_cell_10/mul_2Mul(sequential_6/gru_4/gru_cell_10/add_2:z:0,sequential_6/gru_4/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_6/gru_4/gru_cell_10/IdentityIdentity(sequential_6/gru_4/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_6/gru_4/gru_cell_10/IdentityN	IdentityN(sequential_6/gru_4/gru_cell_10/mul_2:z:0(sequential_6/gru_4/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-136783*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd¬
$sequential_6/gru_4/gru_cell_10/mul_3Mul*sequential_6/gru_4/gru_cell_10/Sigmoid:y:0!sequential_6/gru_4/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
$sequential_6/gru_4/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"sequential_6/gru_4/gru_cell_10/subSub-sequential_6/gru_4/gru_cell_10/sub/x:output:0*sequential_6/gru_4/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
$sequential_6/gru_4/gru_cell_10/mul_4Mul&sequential_6/gru_4/gru_cell_10/sub:z:01sequential_6/gru_4/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
$sequential_6/gru_4/gru_cell_10/add_3AddV2(sequential_6/gru_4/gru_cell_10/mul_3:z:0(sequential_6/gru_4/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential_6/gru_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ñ
"sequential_6/gru_4/TensorArrayV2_1TensorListReserve9sequential_6/gru_4/TensorArrayV2_1/element_shape:output:0+sequential_6/gru_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_6/gru_4/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_6/gru_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_6/gru_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : µ
sequential_6/gru_4/whileWhile.sequential_6/gru_4/while/loop_counter:output:04sequential_6/gru_4/while/maximum_iterations:output:0 sequential_6/gru_4/time:output:0+sequential_6/gru_4/TensorArrayV2_1:handle:0!sequential_6/gru_4/zeros:output:0+sequential_6/gru_4/strided_slice_1:output:0Jsequential_6/gru_4/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_6_gru_4_gru_cell_10_readvariableop_resource=sequential_6_gru_4_gru_cell_10_matmul_readvariableop_resource?sequential_6_gru_4_gru_cell_10_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *0
body(R&
$sequential_6_gru_4_while_body_136799*0
cond(R&
$sequential_6_gru_4_while_cond_136798*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Csequential_6/gru_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   û
5sequential_6/gru_4/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_6/gru_4/while:output:3Lsequential_6/gru_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0{
(sequential_6/gru_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_6/gru_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_6/gru_4/strided_slice_3StridedSlice>sequential_6/gru_4/TensorArrayV2Stack/TensorListStack:tensor:01sequential_6/gru_4/strided_slice_3/stack:output:03sequential_6/gru_4/strided_slice_3/stack_1:output:03sequential_6/gru_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskx
#sequential_6/gru_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_6/gru_4/transpose_1	Transpose>sequential_6/gru_4/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_6/gru_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿddn
sequential_6/gru_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
sequential_6/gru_5/ShapeShape"sequential_6/gru_4/transpose_1:y:0*
T0*
_output_shapes
:p
&sequential_6/gru_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_6/gru_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_6/gru_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_6/gru_5/strided_sliceStridedSlice!sequential_6/gru_5/Shape:output:0/sequential_6/gru_5/strided_slice/stack:output:01sequential_6/gru_5/strided_slice/stack_1:output:01sequential_6/gru_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_6/gru_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¬
sequential_6/gru_5/zeros/packedPack)sequential_6/gru_5/strided_slice:output:0*sequential_6/gru_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_6/gru_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_6/gru_5/zerosFill(sequential_6/gru_5/zeros/packed:output:0'sequential_6/gru_5/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
!sequential_6/gru_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¯
sequential_6/gru_5/transpose	Transpose"sequential_6/gru_4/transpose_1:y:0*sequential_6/gru_5/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿdj
sequential_6/gru_5/Shape_1Shape sequential_6/gru_5/transpose:y:0*
T0*
_output_shapes
:r
(sequential_6/gru_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/gru_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_6/gru_5/strided_slice_1StridedSlice#sequential_6/gru_5/Shape_1:output:01sequential_6/gru_5/strided_slice_1/stack:output:03sequential_6/gru_5/strided_slice_1/stack_1:output:03sequential_6/gru_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_6/gru_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_6/gru_5/TensorArrayV2TensorListReserve7sequential_6/gru_5/TensorArrayV2/element_shape:output:0+sequential_6/gru_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_6/gru_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
:sequential_6/gru_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_6/gru_5/transpose:y:0Qsequential_6/gru_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_6/gru_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_6/gru_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_6/gru_5/strided_slice_2StridedSlice sequential_6/gru_5/transpose:y:01sequential_6/gru_5/strided_slice_2/stack:output:03sequential_6/gru_5/strided_slice_2/stack_1:output:03sequential_6/gru_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_mask¥
-sequential_6/gru_5/gru_cell_11/ReadVariableOpReadVariableOp6sequential_6_gru_5_gru_cell_11_readvariableop_resource*
_output_shapes
:	¬*
dtype0
&sequential_6/gru_5/gru_cell_11/unstackUnpack5sequential_6/gru_5/gru_cell_11/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num³
4sequential_6/gru_5/gru_cell_11/MatMul/ReadVariableOpReadVariableOp=sequential_6_gru_5_gru_cell_11_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Í
%sequential_6/gru_5/gru_cell_11/MatMulMatMul+sequential_6/gru_5/strided_slice_2:output:0<sequential_6/gru_5/gru_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Æ
&sequential_6/gru_5/gru_cell_11/BiasAddBiasAdd/sequential_6/gru_5/gru_cell_11/MatMul:product:0/sequential_6/gru_5/gru_cell_11/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
.sequential_6/gru_5/gru_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$sequential_6/gru_5/gru_cell_11/splitSplit7sequential_6/gru_5/gru_cell_11/split/split_dim:output:0/sequential_6/gru_5/gru_cell_11/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split·
6sequential_6/gru_5/gru_cell_11/MatMul_1/ReadVariableOpReadVariableOp?sequential_6_gru_5_gru_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0Ç
'sequential_6/gru_5/gru_cell_11/MatMul_1MatMul!sequential_6/gru_5/zeros:output:0>sequential_6/gru_5/gru_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ê
(sequential_6/gru_5/gru_cell_11/BiasAdd_1BiasAdd1sequential_6/gru_5/gru_cell_11/MatMul_1:product:0/sequential_6/gru_5/gru_cell_11/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
$sequential_6/gru_5/gru_cell_11/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿ{
0sequential_6/gru_5/gru_cell_11/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&sequential_6/gru_5/gru_cell_11/split_1SplitV1sequential_6/gru_5/gru_cell_11/BiasAdd_1:output:0-sequential_6/gru_5/gru_cell_11/Const:output:09sequential_6/gru_5/gru_cell_11/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split½
"sequential_6/gru_5/gru_cell_11/addAddV2-sequential_6/gru_5/gru_cell_11/split:output:0/sequential_6/gru_5/gru_cell_11/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_6/gru_5/gru_cell_11/SigmoidSigmoid&sequential_6/gru_5/gru_cell_11/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¿
$sequential_6/gru_5/gru_cell_11/add_1AddV2-sequential_6/gru_5/gru_cell_11/split:output:1/sequential_6/gru_5/gru_cell_11/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_6/gru_5/gru_cell_11/Sigmoid_1Sigmoid(sequential_6/gru_5/gru_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdº
"sequential_6/gru_5/gru_cell_11/mulMul,sequential_6/gru_5/gru_cell_11/Sigmoid_1:y:0/sequential_6/gru_5/gru_cell_11/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¶
$sequential_6/gru_5/gru_cell_11/add_2AddV2-sequential_6/gru_5/gru_cell_11/split:output:2&sequential_6/gru_5/gru_cell_11/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
#sequential_6/gru_5/gru_cell_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
$sequential_6/gru_5/gru_cell_11/mul_1Mul,sequential_6/gru_5/gru_cell_11/beta:output:0(sequential_6/gru_5/gru_cell_11/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_6/gru_5/gru_cell_11/Sigmoid_2Sigmoid(sequential_6/gru_5/gru_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
$sequential_6/gru_5/gru_cell_11/mul_2Mul(sequential_6/gru_5/gru_cell_11/add_2:z:0,sequential_6/gru_5/gru_cell_11/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential_6/gru_5/gru_cell_11/IdentityIdentity(sequential_6/gru_5/gru_cell_11/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_6/gru_5/gru_cell_11/IdentityN	IdentityN(sequential_6/gru_5/gru_cell_11/mul_2:z:0(sequential_6/gru_5/gru_cell_11/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-136946*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd¬
$sequential_6/gru_5/gru_cell_11/mul_3Mul*sequential_6/gru_5/gru_cell_11/Sigmoid:y:0!sequential_6/gru_5/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
$sequential_6/gru_5/gru_cell_11/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"sequential_6/gru_5/gru_cell_11/subSub-sequential_6/gru_5/gru_cell_11/sub/x:output:0*sequential_6/gru_5/gru_cell_11/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
$sequential_6/gru_5/gru_cell_11/mul_4Mul&sequential_6/gru_5/gru_cell_11/sub:z:01sequential_6/gru_5/gru_cell_11/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd³
$sequential_6/gru_5/gru_cell_11/add_3AddV2(sequential_6/gru_5/gru_cell_11/mul_3:z:0(sequential_6/gru_5/gru_cell_11/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
0sequential_6/gru_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ñ
"sequential_6/gru_5/TensorArrayV2_1TensorListReserve9sequential_6/gru_5/TensorArrayV2_1/element_shape:output:0+sequential_6/gru_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_6/gru_5/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_6/gru_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_6/gru_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : µ
sequential_6/gru_5/whileWhile.sequential_6/gru_5/while/loop_counter:output:04sequential_6/gru_5/while/maximum_iterations:output:0 sequential_6/gru_5/time:output:0+sequential_6/gru_5/TensorArrayV2_1:handle:0!sequential_6/gru_5/zeros:output:0+sequential_6/gru_5/strided_slice_1:output:0Jsequential_6/gru_5/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_6_gru_5_gru_cell_11_readvariableop_resource=sequential_6_gru_5_gru_cell_11_matmul_readvariableop_resource?sequential_6_gru_5_gru_cell_11_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *0
body(R&
$sequential_6_gru_5_while_body_136962*0
cond(R&
$sequential_6_gru_5_while_cond_136961*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Csequential_6/gru_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   û
5sequential_6/gru_5/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_6/gru_5/while:output:3Lsequential_6/gru_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿd*
element_dtype0{
(sequential_6/gru_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_6/gru_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_6/gru_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_6/gru_5/strided_slice_3StridedSlice>sequential_6/gru_5/TensorArrayV2Stack/TensorListStack:tensor:01sequential_6/gru_5/strided_slice_3/stack:output:03sequential_6/gru_5/strided_slice_3/stack_1:output:03sequential_6/gru_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskx
#sequential_6/gru_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_6/gru_5/transpose_1	Transpose>sequential_6/gru_5/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_6/gru_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿddn
sequential_6/gru_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0¸
sequential_6/dense_6/MatMulMatMul+sequential_6/gru_5/strided_slice_3:output:02sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%sequential_6/dense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
NoOpNoOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp4^sequential_6/gru_3/gru_cell_9/MatMul/ReadVariableOp6^sequential_6/gru_3/gru_cell_9/MatMul_1/ReadVariableOp-^sequential_6/gru_3/gru_cell_9/ReadVariableOp^sequential_6/gru_3/while5^sequential_6/gru_4/gru_cell_10/MatMul/ReadVariableOp7^sequential_6/gru_4/gru_cell_10/MatMul_1/ReadVariableOp.^sequential_6/gru_4/gru_cell_10/ReadVariableOp^sequential_6/gru_4/while5^sequential_6/gru_5/gru_cell_11/MatMul/ReadVariableOp7^sequential_6/gru_5/gru_cell_11/MatMul_1/ReadVariableOp.^sequential_6/gru_5/gru_cell_11/ReadVariableOp^sequential_6/gru_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : 2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2j
3sequential_6/gru_3/gru_cell_9/MatMul/ReadVariableOp3sequential_6/gru_3/gru_cell_9/MatMul/ReadVariableOp2n
5sequential_6/gru_3/gru_cell_9/MatMul_1/ReadVariableOp5sequential_6/gru_3/gru_cell_9/MatMul_1/ReadVariableOp2\
,sequential_6/gru_3/gru_cell_9/ReadVariableOp,sequential_6/gru_3/gru_cell_9/ReadVariableOp24
sequential_6/gru_3/whilesequential_6/gru_3/while2l
4sequential_6/gru_4/gru_cell_10/MatMul/ReadVariableOp4sequential_6/gru_4/gru_cell_10/MatMul/ReadVariableOp2p
6sequential_6/gru_4/gru_cell_10/MatMul_1/ReadVariableOp6sequential_6/gru_4/gru_cell_10/MatMul_1/ReadVariableOp2^
-sequential_6/gru_4/gru_cell_10/ReadVariableOp-sequential_6/gru_4/gru_cell_10/ReadVariableOp24
sequential_6/gru_4/whilesequential_6/gru_4/while2l
4sequential_6/gru_5/gru_cell_11/MatMul/ReadVariableOp4sequential_6/gru_5/gru_cell_11/MatMul/ReadVariableOp2p
6sequential_6/gru_5/gru_cell_11/MatMul_1/ReadVariableOp6sequential_6/gru_5/gru_cell_11/MatMul_1/ReadVariableOp2^
-sequential_6/gru_5/gru_cell_11/ReadVariableOp-sequential_6/gru_5/gru_cell_11/ReadVariableOp24
sequential_6/gru_5/whilesequential_6/gru_5/while:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namegru_3_input
£R

A__inference_gru_4_layer_call_and_return_conditional_losses_138467

inputs6
#gru_cell_10_readvariableop_resource:	¬=
*gru_cell_10_matmul_readvariableop_resource:	d¬?
,gru_cell_10_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_10/MatMul/ReadVariableOp¢#gru_cell_10/MatMul_1/ReadVariableOp¢gru_cell_10/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_10/ReadVariableOpReadVariableOp#gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_10/unstackUnpack"gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_10/MatMul/ReadVariableOpReadVariableOp*gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMulMatMulstrided_slice_2:output:0)gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAddBiasAddgru_cell_10/MatMul:product:0gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_10/splitSplit$gru_cell_10/split/split_dim:output:0gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMul_1MatMulzeros:output:0+gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAdd_1BiasAddgru_cell_10/MatMul_1:product:0gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_10/split_1SplitVgru_cell_10/BiasAdd_1:output:0gru_cell_10/Const:output:0&gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_10/addAddV2gru_cell_10/split:output:0gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_10/SigmoidSigmoidgru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/add_1AddV2gru_cell_10/split:output:1gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_1Sigmoidgru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mulMulgru_cell_10/Sigmoid_1:y:0gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_10/add_2AddV2gru_cell_10/split:output:2gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_10/mul_1Mulgru_cell_10/beta:output:0gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_2Sigmoidgru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_10/mul_2Mulgru_cell_10/add_2:z:0gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/IdentityIdentitygru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_10/IdentityN	IdentityNgru_cell_10/mul_2:z:0gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138355*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_10/mul_3Mulgru_cell_10/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_10/subSubgru_cell_10/sub/x:output:0gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mul_4Mulgru_cell_10/sub:z:0gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_10/add_3AddV2gru_cell_10/mul_3:z:0gru_cell_10/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_10_readvariableop_resource*gru_cell_10_matmul_readvariableop_resource,gru_cell_10_matmul_1_readvariableop_resource*
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
bodyR
while_body_138371*
condR
while_cond_138370*8
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
:ÿÿÿÿÿÿÿÿÿddµ
NoOpNoOp"^gru_cell_10/MatMul/ReadVariableOp$^gru_cell_10/MatMul_1/ReadVariableOp^gru_cell_10/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_10/MatMul/ReadVariableOp!gru_cell_10/MatMul/ReadVariableOp2J
#gru_cell_10/MatMul_1/ReadVariableOp#gru_cell_10/MatMul_1/ReadVariableOp28
gru_cell_10/ReadVariableOpgru_cell_10/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
K
¼	
gru_4_while_body_140232(
$gru_4_while_gru_4_while_loop_counter.
*gru_4_while_gru_4_while_maximum_iterations
gru_4_while_placeholder
gru_4_while_placeholder_1
gru_4_while_placeholder_2'
#gru_4_while_gru_4_strided_slice_1_0c
_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0D
1gru_4_while_gru_cell_10_readvariableop_resource_0:	¬K
8gru_4_while_gru_cell_10_matmul_readvariableop_resource_0:	d¬M
:gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
gru_4_while_identity
gru_4_while_identity_1
gru_4_while_identity_2
gru_4_while_identity_3
gru_4_while_identity_4%
!gru_4_while_gru_4_strided_slice_1a
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensorB
/gru_4_while_gru_cell_10_readvariableop_resource:	¬I
6gru_4_while_gru_cell_10_matmul_readvariableop_resource:	d¬K
8gru_4_while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢-gru_4/while/gru_cell_10/MatMul/ReadVariableOp¢/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp¢&gru_4/while/gru_cell_10/ReadVariableOp
=gru_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ä
/gru_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0gru_4_while_placeholderFgru_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
&gru_4/while/gru_cell_10/ReadVariableOpReadVariableOp1gru_4_while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
gru_4/while/gru_cell_10/unstackUnpack.gru_4/while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num§
-gru_4/while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp8gru_4_while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0Ê
gru_4/while/gru_cell_10/MatMulMatMul6gru_4/while/TensorArrayV2Read/TensorListGetItem:item:05gru_4/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬±
gru_4/while/gru_cell_10/BiasAddBiasAdd(gru_4/while/gru_cell_10/MatMul:product:0(gru_4/while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
'gru_4/while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_4/while/gru_cell_10/splitSplit0gru_4/while/gru_cell_10/split/split_dim:output:0(gru_4/while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split«
/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp:gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0±
 gru_4/while/gru_cell_10/MatMul_1MatMulgru_4_while_placeholder_27gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬µ
!gru_4/while/gru_cell_10/BiasAdd_1BiasAdd*gru_4/while/gru_cell_10/MatMul_1:product:0(gru_4/while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬r
gru_4/while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿt
)gru_4/while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_4/while/gru_cell_10/split_1SplitV*gru_4/while/gru_cell_10/BiasAdd_1:output:0&gru_4/while/gru_cell_10/Const:output:02gru_4/while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split¨
gru_4/while/gru_cell_10/addAddV2&gru_4/while/gru_cell_10/split:output:0(gru_4/while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_4/while/gru_cell_10/SigmoidSigmoidgru_4/while/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
gru_4/while/gru_cell_10/add_1AddV2&gru_4/while/gru_cell_10/split:output:1(gru_4/while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_4/while/gru_cell_10/Sigmoid_1Sigmoid!gru_4/while/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¥
gru_4/while/gru_cell_10/mulMul%gru_4/while/gru_cell_10/Sigmoid_1:y:0(gru_4/while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¡
gru_4/while/gru_cell_10/add_2AddV2&gru_4/while/gru_cell_10/split:output:2gru_4/while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
gru_4/while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ? 
gru_4/while/gru_cell_10/mul_1Mul%gru_4/while/gru_cell_10/beta:output:0!gru_4/while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!gru_4/while/gru_cell_10/Sigmoid_2Sigmoid!gru_4/while/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
gru_4/while/gru_cell_10/mul_2Mul!gru_4/while/gru_cell_10/add_2:z:0%gru_4/while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 gru_4/while/gru_cell_10/IdentityIdentity!gru_4/while/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdë
!gru_4/while/gru_cell_10/IdentityN	IdentityN!gru_4/while/gru_cell_10/mul_2:z:0!gru_4/while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-140282*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
gru_4/while/gru_cell_10/mul_3Mul#gru_4/while/gru_cell_10/Sigmoid:y:0gru_4_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
gru_4/while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_4/while/gru_cell_10/subSub&gru_4/while/gru_cell_10/sub/x:output:0#gru_4/while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd£
gru_4/while/gru_cell_10/mul_4Mulgru_4/while/gru_cell_10/sub:z:0*gru_4/while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_4/while/gru_cell_10/add_3AddV2!gru_4/while/gru_cell_10/mul_3:z:0!gru_4/while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÜ
0gru_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_4_while_placeholder_1gru_4_while_placeholder!gru_4/while/gru_cell_10/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_4/while/addAddV2gru_4_while_placeholdergru_4/while/add/y:output:0*
T0*
_output_shapes
: U
gru_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_4/while/add_1AddV2$gru_4_while_gru_4_while_loop_countergru_4/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_4/while/IdentityIdentitygru_4/while/add_1:z:0^gru_4/while/NoOp*
T0*
_output_shapes
: 
gru_4/while/Identity_1Identity*gru_4_while_gru_4_while_maximum_iterations^gru_4/while/NoOp*
T0*
_output_shapes
: k
gru_4/while/Identity_2Identitygru_4/while/add:z:0^gru_4/while/NoOp*
T0*
_output_shapes
: «
gru_4/while/Identity_3Identity@gru_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_4/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_4/while/Identity_4Identity!gru_4/while/gru_cell_10/add_3:z:0^gru_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÝ
gru_4/while/NoOpNoOp.^gru_4/while/gru_cell_10/MatMul/ReadVariableOp0^gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp'^gru_4/while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_4_while_gru_4_strided_slice_1#gru_4_while_gru_4_strided_slice_1_0"v
8gru_4_while_gru_cell_10_matmul_1_readvariableop_resource:gru_4_while_gru_cell_10_matmul_1_readvariableop_resource_0"r
6gru_4_while_gru_cell_10_matmul_readvariableop_resource8gru_4_while_gru_cell_10_matmul_readvariableop_resource_0"d
/gru_4_while_gru_cell_10_readvariableop_resource1gru_4_while_gru_cell_10_readvariableop_resource_0"5
gru_4_while_identitygru_4/while/Identity:output:0"9
gru_4_while_identity_1gru_4/while/Identity_1:output:0"9
gru_4_while_identity_2gru_4/while/Identity_2:output:0"9
gru_4_while_identity_3gru_4/while/Identity_3:output:0"9
gru_4_while_identity_4gru_4/while/Identity_4:output:0"À
]gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_gru_4_while_tensorarrayv2read_tensorlistgetitem_gru_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2^
-gru_4/while/gru_cell_10/MatMul/ReadVariableOp-gru_4/while/gru_cell_10/MatMul/ReadVariableOp2b
/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp/gru_4/while/gru_cell_10/MatMul_1/ReadVariableOp2P
&gru_4/while/gru_cell_10/ReadVariableOp&gru_4/while/gru_cell_10/ReadVariableOp: 
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
Æ	
ô
C__inference_dense_6_layer_call_and_return_conditional_losses_138659

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
ü

gru_3_while_cond_140068(
$gru_3_while_gru_3_while_loop_counter.
*gru_3_while_gru_3_while_maximum_iterations
gru_3_while_placeholder
gru_3_while_placeholder_1
gru_3_while_placeholder_2*
&gru_3_while_less_gru_3_strided_slice_1@
<gru_3_while_gru_3_while_cond_140068___redundant_placeholder0@
<gru_3_while_gru_3_while_cond_140068___redundant_placeholder1@
<gru_3_while_gru_3_while_cond_140068___redundant_placeholder2@
<gru_3_while_gru_3_while_cond_140068___redundant_placeholder3
gru_3_while_identity
z
gru_3/while/LessLessgru_3_while_placeholder&gru_3_while_less_gru_3_strided_slice_1*
T0*
_output_shapes
: W
gru_3/while/IdentityIdentitygru_3/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_3_while_identitygru_3/while/Identity:output:0*(
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
Ú
ª
while_cond_137342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_137342___redundant_placeholder04
0while_while_cond_137342___redundant_placeholder14
0while_while_cond_137342___redundant_placeholder24
0while_while_cond_137342___redundant_placeholder3
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
!
Ý
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142875

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
:ÿÿÿÿÿÿÿÿÿd£
	IdentityN	IdentityN	mul_2:z:0	add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-142861*:
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


#__inference_internal_grad_fn_144097
result_grads_0
result_grads_1
mul_while_gru_cell_11_beta
mul_while_gru_cell_11_add_2
identity
mulMulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
mul_1Mulmul_while_gru_cell_11_betamul_while_gru_cell_11_add_2*
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
ÿ
·
&__inference_gru_4_layer_call_fn_141271

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallç
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_138467s
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
÷B

while_body_141687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_10_readvariableop_resource_0:	¬E
2while_gru_cell_10_matmul_readvariableop_resource_0:	d¬G
4while_gru_cell_10_matmul_1_readvariableop_resource_0:	d¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_10_readvariableop_resource:	¬C
0while_gru_cell_10_matmul_readvariableop_resource:	d¬E
2while_gru_cell_10_matmul_1_readvariableop_resource:	d¬¢'while/gru_cell_10/MatMul/ReadVariableOp¢)while/gru_cell_10/MatMul_1/ReadVariableOp¢ while/gru_cell_10/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0
 while/gru_cell_10/ReadVariableOpReadVariableOp+while_gru_cell_10_readvariableop_resource_0*
_output_shapes
:	¬*
dtype0
while/gru_cell_10/unstackUnpack(while/gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
'while/gru_cell_10/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0¸
while/gru_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/gru_cell_10/BiasAddBiasAdd"while/gru_cell_10/MatMul:product:0"while/gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
!while/gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_10/splitSplit*while/gru_cell_10/split/split_dim:output:0"while/gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
)while/gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	d¬*
dtype0
while/gru_cell_10/MatMul_1MatMulwhile_placeholder_21while/gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬£
while/gru_cell_10/BiasAdd_1BiasAdd$while/gru_cell_10/MatMul_1:product:0"while/gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬l
while/gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿn
#while/gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_10/split_1SplitV$while/gru_cell_10/BiasAdd_1:output:0 while/gru_cell_10/Const:output:0,while/gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
while/gru_cell_10/addAddV2 while/gru_cell_10/split:output:0"while/gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
while/gru_cell_10/SigmoidSigmoidwhile/gru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_1AddV2 while/gru_cell_10/split:output:1"while/gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_1Sigmoidwhile/gru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mulMulwhile/gru_cell_10/Sigmoid_1:y:0"while/gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_2AddV2 while/gru_cell_10/split:output:2while/gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
while/gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/mul_1Mulwhile/gru_cell_10/beta:output:0while/gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/Sigmoid_2Sigmoidwhile/gru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_2Mulwhile/gru_cell_10/add_2:z:0while/gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
while/gru_cell_10/IdentityIdentitywhile/gru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÙ
while/gru_cell_10/IdentityN	IdentityNwhile/gru_cell_10/mul_2:z:0while/gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141737*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_3Mulwhile/gru_cell_10/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
while/gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_10/subSub while/gru_cell_10/sub/x:output:0while/gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/mul_4Mulwhile/gru_cell_10/sub:z:0$while/gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/gru_cell_10/add_3AddV2while/gru_cell_10/mul_3:z:0while/gru_cell_10/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_10/add_3:z:0*
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
: :éèÒx
while/Identity_4Identitywhile/gru_cell_10/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÅ

while/NoOpNoOp(^while/gru_cell_10/MatMul/ReadVariableOp*^while/gru_cell_10/MatMul_1/ReadVariableOp!^while/gru_cell_10/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_10_matmul_1_readvariableop_resource4while_gru_cell_10_matmul_1_readvariableop_resource_0"f
0while_gru_cell_10_matmul_readvariableop_resource2while_gru_cell_10_matmul_readvariableop_resource_0"X
)while_gru_cell_10_readvariableop_resource+while_gru_cell_10_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿd: : : : : 2R
'while/gru_cell_10/MatMul/ReadVariableOp'while/gru_cell_10/MatMul/ReadVariableOp2V
)while/gru_cell_10/MatMul_1/ReadVariableOp)while/gru_cell_10/MatMul_1/ReadVariableOp2D
 while/gru_cell_10/ReadVariableOp while/gru_cell_10/ReadVariableOp: 
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
Ù

#__inference_internal_grad_fn_143755
result_grads_0
result_grads_1
mul_gru_cell_9_beta
mul_gru_cell_9_add_2
identityx
mulMulmul_gru_cell_9_betamul_gru_cell_9_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
mul_1Mulmul_gru_cell_9_betamul_gru_cell_9_add_2*
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
©
¨
#__inference_internal_grad_fn_143557
result_grads_0
result_grads_1$
 mul_gru_5_while_gru_cell_11_beta%
!mul_gru_5_while_gru_cell_11_add_2
identity
mulMul mul_gru_5_while_gru_cell_11_beta!mul_gru_5_while_gru_cell_11_add_2^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_1Mul mul_gru_5_while_gru_cell_11_beta!mul_gru_5_while_gru_cell_11_add_2*
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
ü

gru_5_while_cond_140394(
$gru_5_while_gru_5_while_loop_counter.
*gru_5_while_gru_5_while_maximum_iterations
gru_5_while_placeholder
gru_5_while_placeholder_1
gru_5_while_placeholder_2*
&gru_5_while_less_gru_5_strided_slice_1@
<gru_5_while_gru_5_while_cond_140394___redundant_placeholder0@
<gru_5_while_gru_5_while_cond_140394___redundant_placeholder1@
<gru_5_while_gru_5_while_cond_140394___redundant_placeholder2@
<gru_5_while_gru_5_while_cond_140394___redundant_placeholder3
gru_5_while_identity
z
gru_5/while/LessLessgru_5_while_placeholder&gru_5_while_less_gru_5_strided_slice_1*
T0*
_output_shapes
: W
gru_5/while/IdentityIdentitygru_5/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_5_while_identitygru_5/while/Identity:output:0*(
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
©
¹
&__inference_gru_4_layer_call_fn_141249
inputs_0
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallò
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
GPU 2J 8 *J
fERC
A__inference_gru_4_layer_call_and_return_conditional_losses_137570|
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

x
#__inference_internal_grad_fn_144061
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
£R

A__inference_gru_4_layer_call_and_return_conditional_losses_139070

inputs6
#gru_cell_10_readvariableop_resource:	¬=
*gru_cell_10_matmul_readvariableop_resource:	d¬?
,gru_cell_10_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_10/MatMul/ReadVariableOp¢#gru_cell_10/MatMul_1/ReadVariableOp¢gru_cell_10/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_10/ReadVariableOpReadVariableOp#gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_10/unstackUnpack"gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_10/MatMul/ReadVariableOpReadVariableOp*gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMulMatMulstrided_slice_2:output:0)gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAddBiasAddgru_cell_10/MatMul:product:0gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_10/splitSplit$gru_cell_10/split/split_dim:output:0gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMul_1MatMulzeros:output:0+gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAdd_1BiasAddgru_cell_10/MatMul_1:product:0gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_10/split_1SplitVgru_cell_10/BiasAdd_1:output:0gru_cell_10/Const:output:0&gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_10/addAddV2gru_cell_10/split:output:0gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_10/SigmoidSigmoidgru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/add_1AddV2gru_cell_10/split:output:1gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_1Sigmoidgru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mulMulgru_cell_10/Sigmoid_1:y:0gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_10/add_2AddV2gru_cell_10/split:output:2gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_10/mul_1Mulgru_cell_10/beta:output:0gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_2Sigmoidgru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_10/mul_2Mulgru_cell_10/add_2:z:0gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/IdentityIdentitygru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_10/IdentityN	IdentityNgru_cell_10/mul_2:z:0gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-138958*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_10/mul_3Mulgru_cell_10/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_10/subSubgru_cell_10/sub/x:output:0gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mul_4Mulgru_cell_10/sub:z:0gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_10/add_3AddV2gru_cell_10/mul_3:z:0gru_cell_10/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_10_readvariableop_resource*gru_cell_10_matmul_readvariableop_resource,gru_cell_10_matmul_1_readvariableop_resource*
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
bodyR
while_body_138974*
condR
while_cond_138973*8
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
:ÿÿÿÿÿÿÿÿÿddµ
NoOpNoOp"^gru_cell_10/MatMul/ReadVariableOp$^gru_cell_10/MatMul_1/ReadVariableOp^gru_cell_10/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_10/MatMul/ReadVariableOp!gru_cell_10/MatMul/ReadVariableOp2J
#gru_cell_10/MatMul_1/ReadVariableOp#gru_cell_10/MatMul_1/ReadVariableOp28
gru_cell_10/ReadVariableOpgru_cell_10/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
£R

A__inference_gru_4_layer_call_and_return_conditional_losses_141950

inputs6
#gru_cell_10_readvariableop_resource:	¬=
*gru_cell_10_matmul_readvariableop_resource:	d¬?
,gru_cell_10_matmul_1_readvariableop_resource:	d¬
identity¢!gru_cell_10/MatMul/ReadVariableOp¢#gru_cell_10/MatMul_1/ReadVariableOp¢gru_cell_10/ReadVariableOp¢while;
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
shrink_axis_mask
gru_cell_10/ReadVariableOpReadVariableOp#gru_cell_10_readvariableop_resource*
_output_shapes
:	¬*
dtype0y
gru_cell_10/unstackUnpack"gru_cell_10/ReadVariableOp:value:0*
T0*"
_output_shapes
:¬:¬*	
num
!gru_cell_10/MatMul/ReadVariableOpReadVariableOp*gru_cell_10_matmul_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMulMatMulstrided_slice_2:output:0)gru_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAddBiasAddgru_cell_10/MatMul:product:0gru_cell_10/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_10/splitSplit$gru_cell_10/split/split_dim:output:0gru_cell_10/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
#gru_cell_10/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	d¬*
dtype0
gru_cell_10/MatMul_1MatMulzeros:output:0+gru_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
gru_cell_10/BiasAdd_1BiasAddgru_cell_10/MatMul_1:product:0gru_cell_10/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬f
gru_cell_10/ConstConst*
_output_shapes
:*
dtype0*!
valueB"d   d   ÿÿÿÿh
gru_cell_10/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_10/split_1SplitVgru_cell_10/BiasAdd_1:output:0gru_cell_10/Const:output:0&gru_cell_10/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
gru_cell_10/addAddV2gru_cell_10/split:output:0gru_cell_10/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
gru_cell_10/SigmoidSigmoidgru_cell_10/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/add_1AddV2gru_cell_10/split:output:1gru_cell_10/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_1Sigmoidgru_cell_10/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mulMulgru_cell_10/Sigmoid_1:y:0gru_cell_10/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
gru_cell_10/add_2AddV2gru_cell_10/split:output:2gru_cell_10/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
gru_cell_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
gru_cell_10/mul_1Mulgru_cell_10/beta:output:0gru_cell_10/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/Sigmoid_2Sigmoidgru_cell_10/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
gru_cell_10/mul_2Mulgru_cell_10/add_2:z:0gru_cell_10/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdi
gru_cell_10/IdentityIdentitygru_cell_10/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÇ
gru_cell_10/IdentityN	IdentityNgru_cell_10/mul_2:z:0gru_cell_10/add_2:z:0*
T
2*,
_gradient_op_typeCustomGradient-141838*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿds
gru_cell_10/mul_3Mulgru_cell_10/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
gru_cell_10/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_10/subSubgru_cell_10/sub/x:output:0gru_cell_10/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
gru_cell_10/mul_4Mulgru_cell_10/sub:z:0gru_cell_10/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
gru_cell_10/add_3AddV2gru_cell_10/mul_3:z:0gru_cell_10/mul_4:z:0*
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
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_10_readvariableop_resource*gru_cell_10_matmul_readvariableop_resource,gru_cell_10_matmul_1_readvariableop_resource*
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
bodyR
while_body_141854*
condR
while_cond_141853*8
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
:ÿÿÿÿÿÿÿÿÿddµ
NoOpNoOp"^gru_cell_10/MatMul/ReadVariableOp$^gru_cell_10/MatMul_1/ReadVariableOp^gru_cell_10/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd: : : 2F
!gru_cell_10/MatMul/ReadVariableOp!gru_cell_10/MatMul/ReadVariableOp2J
#gru_cell_10/MatMul_1/ReadVariableOp#gru_cell_10/MatMul_1/ReadVariableOp28
gru_cell_10/ReadVariableOpgru_cell_10/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
÷
·
&__inference_gru_5_layer_call_fn_141994

inputs
unknown:	¬
	unknown_0:	d¬
	unknown_1:	d¬
identity¢StatefulPartitionedCallã
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
GPU 2J 8 *J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_138881o
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
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_143143CustomGradient-136620<
#__inference_internal_grad_fn_143161CustomGradient-136783<
#__inference_internal_grad_fn_143179CustomGradient-136946<
#__inference_internal_grad_fn_143197CustomGradient-136686<
#__inference_internal_grad_fn_143215CustomGradient-136849<
#__inference_internal_grad_fn_143233CustomGradient-137012<
#__inference_internal_grad_fn_143251CustomGradient-138181<
#__inference_internal_grad_fn_143269CustomGradient-138247<
#__inference_internal_grad_fn_143287CustomGradient-138355<
#__inference_internal_grad_fn_143305CustomGradient-138421<
#__inference_internal_grad_fn_143323CustomGradient-138529<
#__inference_internal_grad_fn_143341CustomGradient-138595<
#__inference_internal_grad_fn_143359CustomGradient-139147<
#__inference_internal_grad_fn_143377CustomGradient-139213<
#__inference_internal_grad_fn_143395CustomGradient-138958<
#__inference_internal_grad_fn_143413CustomGradient-139024<
#__inference_internal_grad_fn_143431CustomGradient-138769<
#__inference_internal_grad_fn_143449CustomGradient-138835<
#__inference_internal_grad_fn_143467CustomGradient-139554<
#__inference_internal_grad_fn_143485CustomGradient-139717<
#__inference_internal_grad_fn_143503CustomGradient-139880<
#__inference_internal_grad_fn_143521CustomGradient-139620<
#__inference_internal_grad_fn_143539CustomGradient-139783<
#__inference_internal_grad_fn_143557CustomGradient-139946<
#__inference_internal_grad_fn_143575CustomGradient-140053<
#__inference_internal_grad_fn_143593CustomGradient-140216<
#__inference_internal_grad_fn_143611CustomGradient-140379<
#__inference_internal_grad_fn_143629CustomGradient-140119<
#__inference_internal_grad_fn_143647CustomGradient-140282<
#__inference_internal_grad_fn_143665CustomGradient-140445<
#__inference_internal_grad_fn_143683CustomGradient-137127<
#__inference_internal_grad_fn_143701CustomGradient-137277<
#__inference_internal_grad_fn_143719CustomGradient-140625<
#__inference_internal_grad_fn_143737CustomGradient-140691<
#__inference_internal_grad_fn_143755CustomGradient-140792<
#__inference_internal_grad_fn_143773CustomGradient-140858<
#__inference_internal_grad_fn_143791CustomGradient-140959<
#__inference_internal_grad_fn_143809CustomGradient-141025<
#__inference_internal_grad_fn_143827CustomGradient-141126<
#__inference_internal_grad_fn_143845CustomGradient-141192<
#__inference_internal_grad_fn_143863CustomGradient-137479<
#__inference_internal_grad_fn_143881CustomGradient-137629<
#__inference_internal_grad_fn_143899CustomGradient-141337<
#__inference_internal_grad_fn_143917CustomGradient-141403<
#__inference_internal_grad_fn_143935CustomGradient-141504<
#__inference_internal_grad_fn_143953CustomGradient-141570<
#__inference_internal_grad_fn_143971CustomGradient-141671<
#__inference_internal_grad_fn_143989CustomGradient-141737<
#__inference_internal_grad_fn_144007CustomGradient-141838<
#__inference_internal_grad_fn_144025CustomGradient-141904<
#__inference_internal_grad_fn_144043CustomGradient-137831<
#__inference_internal_grad_fn_144061CustomGradient-137981<
#__inference_internal_grad_fn_144079CustomGradient-142049<
#__inference_internal_grad_fn_144097CustomGradient-142115<
#__inference_internal_grad_fn_144115CustomGradient-142216<
#__inference_internal_grad_fn_144133CustomGradient-142282<
#__inference_internal_grad_fn_144151CustomGradient-142383<
#__inference_internal_grad_fn_144169CustomGradient-142449<
#__inference_internal_grad_fn_144187CustomGradient-142550<
#__inference_internal_grad_fn_144205CustomGradient-142616<
#__inference_internal_grad_fn_144223CustomGradient-142741<
#__inference_internal_grad_fn_144241CustomGradient-142787<
#__inference_internal_grad_fn_144259CustomGradient-142861<
#__inference_internal_grad_fn_144277CustomGradient-142907<
#__inference_internal_grad_fn_144295CustomGradient-142981<
#__inference_internal_grad_fn_144313CustomGradient-143027"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
G
gru_3_input8
serving_default_gru_3_input:0ÿÿÿÿÿÿÿÿÿd;
dense_60
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ªõ
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
2ÿ
-__inference_sequential_6_layer_call_fn_138691
-__inference_sequential_6_layer_call_fn_139472
-__inference_sequential_6_layer_call_fn_139499
-__inference_sequential_6_layer_call_fn_139379À
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
î2ë
H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
H__inference_sequential_6_layer_call_and_return_conditional_losses_139409
H__inference_sequential_6_layer_call_and_return_conditional_losses_139439À
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
ÐBÍ
!__inference__wrapped_model_137064gru_3_input"
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
û2ø
&__inference_gru_3_layer_call_fn_140537
&__inference_gru_3_layer_call_fn_140548
&__inference_gru_3_layer_call_fn_140559
&__inference_gru_3_layer_call_fn_140570Õ
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
ç2ä
A__inference_gru_3_layer_call_and_return_conditional_losses_140737
A__inference_gru_3_layer_call_and_return_conditional_losses_140904
A__inference_gru_3_layer_call_and_return_conditional_losses_141071
A__inference_gru_3_layer_call_and_return_conditional_losses_141238Õ
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
û2ø
&__inference_gru_4_layer_call_fn_141249
&__inference_gru_4_layer_call_fn_141260
&__inference_gru_4_layer_call_fn_141271
&__inference_gru_4_layer_call_fn_141282Õ
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
ç2ä
A__inference_gru_4_layer_call_and_return_conditional_losses_141449
A__inference_gru_4_layer_call_and_return_conditional_losses_141616
A__inference_gru_4_layer_call_and_return_conditional_losses_141783
A__inference_gru_4_layer_call_and_return_conditional_losses_141950Õ
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
û2ø
&__inference_gru_5_layer_call_fn_141961
&__inference_gru_5_layer_call_fn_141972
&__inference_gru_5_layer_call_fn_141983
&__inference_gru_5_layer_call_fn_141994Õ
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
ç2ä
A__inference_gru_5_layer_call_and_return_conditional_losses_142161
A__inference_gru_5_layer_call_and_return_conditional_losses_142328
A__inference_gru_5_layer_call_and_return_conditional_losses_142495
A__inference_gru_5_layer_call_and_return_conditional_losses_142662Õ
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
 :d2dense_6/kernel
:2dense_6/bias
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
Ò2Ï
(__inference_dense_6_layer_call_fn_142671¢
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
í2ê
C__inference_dense_6_layer_call_and_return_conditional_losses_142681¢
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
*:(	¬2gru_3/gru_cell_9/kernel
4:2	d¬2!gru_3/gru_cell_9/recurrent_kernel
(:&	¬2gru_3/gru_cell_9/bias
+:)	d¬2gru_4/gru_cell_10/kernel
5:3	d¬2"gru_4/gru_cell_10/recurrent_kernel
):'	¬2gru_4/gru_cell_10/bias
+:)	d¬2gru_5/gru_cell_11/kernel
5:3	d¬2"gru_5/gru_cell_11/recurrent_kernel
):'	¬2gru_5/gru_cell_11/bias
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
ÏBÌ
$__inference_signature_wrapper_140526gru_3_input"
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
2
+__inference_gru_cell_9_layer_call_fn_142695
+__inference_gru_cell_9_layer_call_fn_142709¾
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
Ô2Ñ
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142755
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142801¾
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
 2
,__inference_gru_cell_10_layer_call_fn_142815
,__inference_gru_cell_10_layer_call_fn_142829¾
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
Ö2Ó
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142875
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142921¾
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
 2
,__inference_gru_cell_11_layer_call_fn_142935
,__inference_gru_cell_11_layer_call_fn_142949¾
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
Ö2Ó
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_142995
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_143041¾
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
%:#d2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
/:-	¬2Adam/gru_3/gru_cell_9/kernel/m
9:7	d¬2(Adam/gru_3/gru_cell_9/recurrent_kernel/m
-:+	¬2Adam/gru_3/gru_cell_9/bias/m
0:.	d¬2Adam/gru_4/gru_cell_10/kernel/m
::8	d¬2)Adam/gru_4/gru_cell_10/recurrent_kernel/m
.:,	¬2Adam/gru_4/gru_cell_10/bias/m
0:.	d¬2Adam/gru_5/gru_cell_11/kernel/m
::8	d¬2)Adam/gru_5/gru_cell_11/recurrent_kernel/m
.:,	¬2Adam/gru_5/gru_cell_11/bias/m
%:#d2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
/:-	¬2Adam/gru_3/gru_cell_9/kernel/v
9:7	d¬2(Adam/gru_3/gru_cell_9/recurrent_kernel/v
-:+	¬2Adam/gru_3/gru_cell_9/bias/v
0:.	d¬2Adam/gru_4/gru_cell_10/kernel/v
::8	d¬2)Adam/gru_4/gru_cell_10/recurrent_kernel/v
.:,	¬2Adam/gru_4/gru_cell_10/bias/v
0:.	d¬2Adam/gru_5/gru_cell_11/kernel/v
::8	d¬2)Adam/gru_5/gru_cell_11/recurrent_kernel/v
.:,	¬2Adam/gru_5/gru_cell_11/bias/v
KbI
$sequential_6/gru_3/gru_cell_9/beta:0!__inference__wrapped_model_137064
LbJ
%sequential_6/gru_3/gru_cell_9/add_2:0!__inference__wrapped_model_137064
LbJ
%sequential_6/gru_4/gru_cell_10/beta:0!__inference__wrapped_model_137064
MbK
&sequential_6/gru_4/gru_cell_10/add_2:0!__inference__wrapped_model_137064
LbJ
%sequential_6/gru_5/gru_cell_11/beta:0!__inference__wrapped_model_137064
MbK
&sequential_6/gru_5/gru_cell_11/add_2:0!__inference__wrapped_model_137064
TbR
*sequential_6/gru_3/while/gru_cell_9/beta:0$sequential_6_gru_3_while_body_136636
UbS
+sequential_6/gru_3/while/gru_cell_9/add_2:0$sequential_6_gru_3_while_body_136636
UbS
+sequential_6/gru_4/while/gru_cell_10/beta:0$sequential_6_gru_4_while_body_136799
VbT
,sequential_6/gru_4/while/gru_cell_10/add_2:0$sequential_6_gru_4_while_body_136799
UbS
+sequential_6/gru_5/while/gru_cell_11/beta:0$sequential_6_gru_5_while_body_136962
VbT
,sequential_6/gru_5/while/gru_cell_11/add_2:0$sequential_6_gru_5_while_body_136962
XbV
gru_cell_9/beta:0A__inference_gru_3_layer_call_and_return_conditional_losses_138293
YbW
gru_cell_9/add_2:0A__inference_gru_3_layer_call_and_return_conditional_losses_138293
.b,
while/gru_cell_9/beta:0while_body_138197
/b-
while/gru_cell_9/add_2:0while_body_138197
YbW
gru_cell_10/beta:0A__inference_gru_4_layer_call_and_return_conditional_losses_138467
ZbX
gru_cell_10/add_2:0A__inference_gru_4_layer_call_and_return_conditional_losses_138467
/b-
while/gru_cell_10/beta:0while_body_138371
0b.
while/gru_cell_10/add_2:0while_body_138371
YbW
gru_cell_11/beta:0A__inference_gru_5_layer_call_and_return_conditional_losses_138641
ZbX
gru_cell_11/add_2:0A__inference_gru_5_layer_call_and_return_conditional_losses_138641
/b-
while/gru_cell_11/beta:0while_body_138545
0b.
while/gru_cell_11/add_2:0while_body_138545
XbV
gru_cell_9/beta:0A__inference_gru_3_layer_call_and_return_conditional_losses_139259
YbW
gru_cell_9/add_2:0A__inference_gru_3_layer_call_and_return_conditional_losses_139259
.b,
while/gru_cell_9/beta:0while_body_139163
/b-
while/gru_cell_9/add_2:0while_body_139163
YbW
gru_cell_10/beta:0A__inference_gru_4_layer_call_and_return_conditional_losses_139070
ZbX
gru_cell_10/add_2:0A__inference_gru_4_layer_call_and_return_conditional_losses_139070
/b-
while/gru_cell_10/beta:0while_body_138974
0b.
while/gru_cell_10/add_2:0while_body_138974
YbW
gru_cell_11/beta:0A__inference_gru_5_layer_call_and_return_conditional_losses_138881
ZbX
gru_cell_11/add_2:0A__inference_gru_5_layer_call_and_return_conditional_losses_138881
/b-
while/gru_cell_11/beta:0while_body_138785
0b.
while/gru_cell_11/add_2:0while_body_138785
ebc
gru_3/gru_cell_9/beta:0H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
fbd
gru_3/gru_cell_9/add_2:0H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
fbd
gru_4/gru_cell_10/beta:0H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
gbe
gru_4/gru_cell_10/add_2:0H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
fbd
gru_5/gru_cell_11/beta:0H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
gbe
gru_5/gru_cell_11/add_2:0H__inference_sequential_6_layer_call_and_return_conditional_losses_139998
:b8
gru_3/while/gru_cell_9/beta:0gru_3_while_body_139570
;b9
gru_3/while/gru_cell_9/add_2:0gru_3_while_body_139570
;b9
gru_4/while/gru_cell_10/beta:0gru_4_while_body_139733
<b:
gru_4/while/gru_cell_10/add_2:0gru_4_while_body_139733
;b9
gru_5/while/gru_cell_11/beta:0gru_5_while_body_139896
<b:
gru_5/while/gru_cell_11/add_2:0gru_5_while_body_139896
ebc
gru_3/gru_cell_9/beta:0H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
fbd
gru_3/gru_cell_9/add_2:0H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
fbd
gru_4/gru_cell_10/beta:0H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
gbe
gru_4/gru_cell_10/add_2:0H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
fbd
gru_5/gru_cell_11/beta:0H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
gbe
gru_5/gru_cell_11/add_2:0H__inference_sequential_6_layer_call_and_return_conditional_losses_140497
:b8
gru_3/while/gru_cell_9/beta:0gru_3_while_body_140069
;b9
gru_3/while/gru_cell_9/add_2:0gru_3_while_body_140069
;b9
gru_4/while/gru_cell_10/beta:0gru_4_while_body_140232
<b:
gru_4/while/gru_cell_10/add_2:0gru_4_while_body_140232
;b9
gru_5/while/gru_cell_11/beta:0gru_5_while_body_140395
<b:
gru_5/while/gru_cell_11/add_2:0gru_5_while_body_140395
RbP
beta:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137141
SbQ
add_2:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137141
RbP
beta:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137291
SbQ
add_2:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_137291
XbV
gru_cell_9/beta:0A__inference_gru_3_layer_call_and_return_conditional_losses_140737
YbW
gru_cell_9/add_2:0A__inference_gru_3_layer_call_and_return_conditional_losses_140737
.b,
while/gru_cell_9/beta:0while_body_140641
/b-
while/gru_cell_9/add_2:0while_body_140641
XbV
gru_cell_9/beta:0A__inference_gru_3_layer_call_and_return_conditional_losses_140904
YbW
gru_cell_9/add_2:0A__inference_gru_3_layer_call_and_return_conditional_losses_140904
.b,
while/gru_cell_9/beta:0while_body_140808
/b-
while/gru_cell_9/add_2:0while_body_140808
XbV
gru_cell_9/beta:0A__inference_gru_3_layer_call_and_return_conditional_losses_141071
YbW
gru_cell_9/add_2:0A__inference_gru_3_layer_call_and_return_conditional_losses_141071
.b,
while/gru_cell_9/beta:0while_body_140975
/b-
while/gru_cell_9/add_2:0while_body_140975
XbV
gru_cell_9/beta:0A__inference_gru_3_layer_call_and_return_conditional_losses_141238
YbW
gru_cell_9/add_2:0A__inference_gru_3_layer_call_and_return_conditional_losses_141238
.b,
while/gru_cell_9/beta:0while_body_141142
/b-
while/gru_cell_9/add_2:0while_body_141142
SbQ
beta:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137493
TbR
add_2:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137493
SbQ
beta:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137643
TbR
add_2:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_137643
YbW
gru_cell_10/beta:0A__inference_gru_4_layer_call_and_return_conditional_losses_141449
ZbX
gru_cell_10/add_2:0A__inference_gru_4_layer_call_and_return_conditional_losses_141449
/b-
while/gru_cell_10/beta:0while_body_141353
0b.
while/gru_cell_10/add_2:0while_body_141353
YbW
gru_cell_10/beta:0A__inference_gru_4_layer_call_and_return_conditional_losses_141616
ZbX
gru_cell_10/add_2:0A__inference_gru_4_layer_call_and_return_conditional_losses_141616
/b-
while/gru_cell_10/beta:0while_body_141520
0b.
while/gru_cell_10/add_2:0while_body_141520
YbW
gru_cell_10/beta:0A__inference_gru_4_layer_call_and_return_conditional_losses_141783
ZbX
gru_cell_10/add_2:0A__inference_gru_4_layer_call_and_return_conditional_losses_141783
/b-
while/gru_cell_10/beta:0while_body_141687
0b.
while/gru_cell_10/add_2:0while_body_141687
YbW
gru_cell_10/beta:0A__inference_gru_4_layer_call_and_return_conditional_losses_141950
ZbX
gru_cell_10/add_2:0A__inference_gru_4_layer_call_and_return_conditional_losses_141950
/b-
while/gru_cell_10/beta:0while_body_141854
0b.
while/gru_cell_10/add_2:0while_body_141854
SbQ
beta:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137845
TbR
add_2:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137845
SbQ
beta:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137995
TbR
add_2:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_137995
YbW
gru_cell_11/beta:0A__inference_gru_5_layer_call_and_return_conditional_losses_142161
ZbX
gru_cell_11/add_2:0A__inference_gru_5_layer_call_and_return_conditional_losses_142161
/b-
while/gru_cell_11/beta:0while_body_142065
0b.
while/gru_cell_11/add_2:0while_body_142065
YbW
gru_cell_11/beta:0A__inference_gru_5_layer_call_and_return_conditional_losses_142328
ZbX
gru_cell_11/add_2:0A__inference_gru_5_layer_call_and_return_conditional_losses_142328
/b-
while/gru_cell_11/beta:0while_body_142232
0b.
while/gru_cell_11/add_2:0while_body_142232
YbW
gru_cell_11/beta:0A__inference_gru_5_layer_call_and_return_conditional_losses_142495
ZbX
gru_cell_11/add_2:0A__inference_gru_5_layer_call_and_return_conditional_losses_142495
/b-
while/gru_cell_11/beta:0while_body_142399
0b.
while/gru_cell_11/add_2:0while_body_142399
YbW
gru_cell_11/beta:0A__inference_gru_5_layer_call_and_return_conditional_losses_142662
ZbX
gru_cell_11/add_2:0A__inference_gru_5_layer_call_and_return_conditional_losses_142662
/b-
while/gru_cell_11/beta:0while_body_142566
0b.
while/gru_cell_11/add_2:0while_body_142566
RbP
beta:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142755
SbQ
add_2:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142755
RbP
beta:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142801
SbQ
add_2:0F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142801
SbQ
beta:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142875
TbR
add_2:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142875
SbQ
beta:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142921
TbR
add_2:0G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142921
SbQ
beta:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_142995
TbR
add_2:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_142995
SbQ
beta:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_143041
TbR
add_2:0G__inference_gru_cell_11_layer_call_and_return_conditional_losses_143041
!__inference__wrapped_model_137064z867;9:><=)*8¢5
.¢+
)&
gru_3_inputÿÿÿÿÿÿÿÿÿd
ª "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_6_layer_call_and_return_conditional_losses_142681\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_6_layer_call_fn_142671O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿÐ
A__inference_gru_3_layer_call_and_return_conditional_losses_140737867O¢L
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
 Ð
A__inference_gru_3_layer_call_and_return_conditional_losses_140904867O¢L
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
 ¶
A__inference_gru_3_layer_call_and_return_conditional_losses_141071q867?¢<
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
 ¶
A__inference_gru_3_layer_call_and_return_conditional_losses_141238q867?¢<
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
 §
&__inference_gru_3_layer_call_fn_140537}867O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd§
&__inference_gru_3_layer_call_fn_140548}867O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
&__inference_gru_3_layer_call_fn_140559d867?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿdd
&__inference_gru_3_layer_call_fn_140570d867?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿddÐ
A__inference_gru_4_layer_call_and_return_conditional_losses_141449;9:O¢L
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
 Ð
A__inference_gru_4_layer_call_and_return_conditional_losses_141616;9:O¢L
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
 ¶
A__inference_gru_4_layer_call_and_return_conditional_losses_141783q;9:?¢<
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
 ¶
A__inference_gru_4_layer_call_and_return_conditional_losses_141950q;9:?¢<
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
 §
&__inference_gru_4_layer_call_fn_141249};9:O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd§
&__inference_gru_4_layer_call_fn_141260};9:O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd
&__inference_gru_4_layer_call_fn_141271d;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿdd
&__inference_gru_4_layer_call_fn_141282d;9:?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿddÂ
A__inference_gru_5_layer_call_and_return_conditional_losses_142161}><=O¢L
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
 Â
A__inference_gru_5_layer_call_and_return_conditional_losses_142328}><=O¢L
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
 ²
A__inference_gru_5_layer_call_and_return_conditional_losses_142495m><=?¢<
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
 ²
A__inference_gru_5_layer_call_and_return_conditional_losses_142662m><=?¢<
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
 
&__inference_gru_5_layer_call_fn_141961p><=O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_gru_5_layer_call_fn_141972p><=O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_gru_5_layer_call_fn_141983`><=?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_gru_5_layer_call_fn_141994`><=?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿdd

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142875·;9:\¢Y
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
 
G__inference_gru_cell_10_layer_call_and_return_conditional_losses_142921·;9:\¢Y
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
 Ú
,__inference_gru_cell_10_layer_call_fn_142815©;9:\¢Y
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
1/0ÿÿÿÿÿÿÿÿÿdÚ
,__inference_gru_cell_10_layer_call_fn_142829©;9:\¢Y
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
1/0ÿÿÿÿÿÿÿÿÿd
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_142995·><=\¢Y
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
 
G__inference_gru_cell_11_layer_call_and_return_conditional_losses_143041·><=\¢Y
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
 Ú
,__inference_gru_cell_11_layer_call_fn_142935©><=\¢Y
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
1/0ÿÿÿÿÿÿÿÿÿdÚ
,__inference_gru_cell_11_layer_call_fn_142949©><=\¢Y
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
1/0ÿÿÿÿÿÿÿÿÿd
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142755·867\¢Y
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
 
F__inference_gru_cell_9_layer_call_and_return_conditional_losses_142801·867\¢Y
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
 Ù
+__inference_gru_cell_9_layer_call_fn_142695©867\¢Y
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
1/0ÿÿÿÿÿÿÿÿÿdÙ
+__inference_gru_cell_9_layer_call_fn_142709©867\¢Y
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
1/0ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143143e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143161e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143179 e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143197¡¢e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143215£¤e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143233¥¦e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143251§¨e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143269©ªe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143287«¬e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143305­®e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143323¯°e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143341±²e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143359³´e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143377µ¶e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143395·¸e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143413¹ºe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143431»¼e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143449½¾e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143467¿Àe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143485ÁÂe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143503ÃÄe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143521ÅÆe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143539ÇÈe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143557ÉÊe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143575ËÌe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143593ÍÎe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143611ÏÐe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143629ÑÒe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143647ÓÔe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143665ÕÖe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143683×Øe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143701ÙÚe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143719ÛÜe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143737ÝÞe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143755ßàe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143773áâe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143791ãäe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143809åæe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143827çèe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143845éêe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143863ëìe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143881íîe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143899ïðe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143917ñòe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143935óôe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143953õöe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143971÷øe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_143989ùúe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144007ûüe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144025ýþe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144043ÿe¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144061e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144079e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144097e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144115e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144133e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144151e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144169e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144187e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144205e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144223e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144241e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144259e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144277e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144295e¢b
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
1ÿÿÿÿÿÿÿÿÿd»
#__inference_internal_grad_fn_144313e¢b
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
1ÿÿÿÿÿÿÿÿÿdÂ
H__inference_sequential_6_layer_call_and_return_conditional_losses_139409v867;9:><=)*@¢=
6¢3
)&
gru_3_inputÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
H__inference_sequential_6_layer_call_and_return_conditional_losses_139439v867;9:><=)*@¢=
6¢3
)&
gru_3_inputÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
H__inference_sequential_6_layer_call_and_return_conditional_losses_139998q867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
H__inference_sequential_6_layer_call_and_return_conditional_losses_140497q867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_6_layer_call_fn_138691i867;9:><=)*@¢=
6¢3
)&
gru_3_inputÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_6_layer_call_fn_139379i867;9:><=)*@¢=
6¢3
)&
gru_3_inputÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_6_layer_call_fn_139472d867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_6_layer_call_fn_139499d867;9:><=)*;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd
p

 
ª "ÿÿÿÿÿÿÿÿÿ²
$__inference_signature_wrapper_140526867;9:><=)*G¢D
¢ 
=ª:
8
gru_3_input)&
gru_3_inputÿÿÿÿÿÿÿÿÿd"1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ