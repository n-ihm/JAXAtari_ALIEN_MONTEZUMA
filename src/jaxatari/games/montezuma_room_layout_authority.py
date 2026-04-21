from jaxatari.games.jax_mzuma_utils import SANTAH, DYNAMIC, STATIC, CONSTANT_SHAPE, ROOM_SHAPED, NAMED_TUPLE_STACK, SINGLETON_INT, TagNamedTuple, VanillaRoom, RoomConnectionObject, RequiredRoomFields, NamedTupleFieldType
from typing import NamedTuple, get_type_hints, Annotated, get_origin, get_args, Optional, Type, Callable, Dict, Any
from enum import Enum
import warnings
from jaxatari.games.jax_montezuma_constants import *
from jax import Array as jArray
import jax.numpy as jnp
from jaxatari.games.jax_mzuma_utils import PyramidLayout


class NewRoom:
    def __init__(self, room_id: int, underlyingTupleClass: NamedTuple):
        """Initialization method for the "Room" handler class. 
           At its core, it wraps around a tuple of the underlyingTupleClass
           and provides a convenient wrapper around individual instances of the
           Named Tuple. Rooms are automatically tracked by the layout manager & integrated with the 
           persistence infrastructure. 
           
           Provides the option to set individual fields to Arrays, Lists of Scalar valued Integer Named tuples, 
           and singleton arrays. 
           ScalarValuedIntegerNamedTuples need to always be registered at the ScalarNamedIntegerTupleDeserialisaitonHandler first.

        Args:
            room_id (int): ID of the underlying Room
            underlyingTupleClass (NamedTuple): NamedTuple Class that gives the basis for the room-tuple this object wraps around.
        """
        self.underlyingTupleClass: NamedTuple = underlyingTupleClass
        self.room_id: int = room_id
        self.connection_object: RoomConnectionObject = RoomConnectionObject(room_id=self.room_id)
        self.field_contents: Dict[str, Any] = {}
        self.field_contents[RequiredRoomFields.ROOM_ID.value] = jnp.array([self.room_id], dtype=jnp.uint16)
        self.field_persistent: Dict[str, bool] = {} # Whether a certain field is supposed to be stored in the 
            # persistant global storage. 
        # Room-IDs may never change, so they don't require persistence.
        self.field_persistent[RequiredRoomFields.ROOM_ID.value] = False
        self.field_type: Dict[str, NamedTupleFieldType] = {} # type of content in this field.
        self.field_type[RequiredRoomFields.ROOM_ID.value] = NamedTupleFieldType.INTEGER_SCALAR
        self.field_named_tuples: Dict[str, NamedTuple] = {} # for all fields that are supposed to be synchronized to 
            # the required global storage and have a namedtuple as value, the class of the named_tuple is stored in here. 
        self.present_fields: List[str] = list(self.underlyingTupleClass._fields)
        if not RequiredRoomFields.ROOM_ID.value in self.present_fields:
            raise Exception("All NamedTuples representing individual rooms are required to have a 'ROOM_ID' field")
        
        
        
    def set_field(self, field_name: str, field_type: NamedTupleFieldType, content: jnp.ndarray|List[NamedTuple], requires_serialisation: bool = False, named_tuple_type: NamedTuple=None):
        """The main method through which fields of individual rooms are set.
        Args:
            field_name (str): name of the field for which the value is to be set. 
                This field name is required to actually be a field in the given namedtuple subclass.
            field_type (NamedTupleFieldType): Type of the field to be set. This can either be integer singleton array, 
                arbitrary array, or a list of NamedTuples which are eventually stored as an array. 
                If it is a list of named tuples, all need to have the same type.
            content (jnp.ndarray | List[NamedTuple]): Content of the field as described above.
            requires_serialisation (bool, optional): Whether the field is supposed to be serialised to the global state
                CAUTION: ONLY SINGLETON INTEGER FIELDS OR NAMED TUPLE FIELDS SUPPORT SYNCHRONISATION TO THE GLOBAL STATE. Defaults to False.
        """
        # Check whether valid field content was passed.
        if not field_name in self.present_fields:
            raise Exception("Attempted to add content to a field that is not present in the underlying namedtuple")
        if field_name == RequiredRoomFields.ROOM_ID.value:
            raise Exception("Cannot overwrite the fixed field 'ROOM_ID'")
        if field_type == NamedTupleFieldType.INTEGER_SCALAR:
            if not isinstance(content, jArray):
                raise Exception("Missmatch between stated field_type and given field_type")
            data: jnp.ndarray = content
            if not data.dtype == jnp.int32:
                raise Exception("Cannot support non int32 singleton arrays")
            if data.shape != (1,):
                raise Exception("Only support singleton arrays of shape '(1, )'")
            self.field_contents[field_name] = data
            self.field_persistent[field_name] = requires_serialisation
            self.field_type[field_name] = NamedTupleFieldType.INTEGER_SCALAR
            
        elif field_type == NamedTupleFieldType.NAMED_TUPLE_STACK:
            if named_tuple_type is None:
                raise Exception("If a list of named tuples is provided as field content, a NamedTupleType needs to be given as well")
            if not isinstance(content, List):
                raise Exception("Expected a list of NamedTupleObjects as content")
            for i in content:
                if not isinstance(i, named_tuple_type):
                    raise Exception("All items in the list need to be of the required named_tuple_type")
            if named_tuple_type not in SANTAH.full_serialisations:
                raise Exception("The NamedTupleType need to be registered with the ScalarNamedTupleDeserialisationHandler")
            field_content: jArray = jnp.zeros((len(content), SANTAH.fully_serialised_sizes[named_tuple_type]), dtype=jnp.int32)
            for i, d in enumerate(content):
                serialised = SANTAH.full_serialisations[named_tuple_type](d)
                field_content = field_content.at[i, ...].set(serialised)
            self.field_contents[field_name] = field_content
            self.field_persistent[field_name] = requires_serialisation
            self.field_type[field_name] = NamedTupleFieldType.NAMED_TUPLE_STACK
            self.field_named_tuples[field_name] = named_tuple_type
            
        elif field_type == NamedTupleFieldType.OTHER_ARRAY:
            if requires_serialisation:
                raise Exception("Cannot serialise field of type 'OTHER_ARRAY'")
            self.field_contents[field_name] = content
            self.field_persistent[field_name] = False
            self.field_type[field_name] = NamedTupleFieldType.OTHER_ARRAY

        elif field_type == NamedTupleFieldType.ROOM_SIZED_ARRAY:
            if requires_serialisation:
                raise Exception("Cannot serialise field of type 'ROOM_SIZED_ARRAY'")
            if not isinstance(content, jArray):
                raise Exception("ROOM_SIZED_ARRAY content must be a jax/numpy array")
            if content.ndim < 2:
                raise Exception("ROOM_SIZED_ARRAY fields must have at least 2 dimensions (dim 1 is the room-height dimension)")
            self.field_contents[field_name] = content
            self.field_persistent[field_name] = False
            self.field_type[field_name] = NamedTupleFieldType.ROOM_SIZED_ARRAY
            
            
    def get_jitted_room_constructor(self)->Callable[[], NamedTuple]:
        """Returns a jitted function that constructs the room-namedtuple specified by this object.

        Returns:
            Callable[[], NamedTuple]: _description_
        """
        
        # Collect all the fields that need to be constructed according to the Tags this room implements.
        implemented_tags: Tuple[Enum] = SANTAH.room_tags[self.underlyingTupleClass]
        tag_constructed_fields = set([])
        for tag in implemented_tags:
            if tag in SANTAH.tag_based_room_constructor_fields:
                constructed_fields = SANTAH.tag_based_room_constructor_fields[tag].keys()
                tag_constructed_fields = tag_constructed_fields.union(set(constructed_fields))
                

        if set(self.field_contents.keys()).union(list(SANTAH._fields_constructor[self.underlyingTupleClass].keys())).union(
            
            list(SANTAH._vanilla_room_field_constructors.keys())).union(tag_constructed_fields) != set(self.present_fields):
            # Fields for which an explicit constructor is given are not required to be
            # initialized manually
            # Explicit constructor can either be given at the proto room level, at the tag level or at the per-room level. 
            # Per room level takes precedence over proto room level & the tag level
            raise Exception("Not all fields are set. All fields need to be set before generating infrastructure functions.")
        # For all constructed fields for which an init value is used, the init value is used 
        # to initialize the respective fields of the named tuple used to construct the default fields. 
        # For all remaining fields, a default value is used.
        
        if SANTAH.vanilla_room is None:
            raise Exception("Vanilla Room has not been set.")
        for k in list(SANTAH._fields_constructor[self.underlyingTupleClass].keys()):
            if k in self.field_contents:
                if self.field_type[k] != NamedTupleFieldType.OTHER_ARRAY:
                    raise Exception("Fields for which an explicit constructor is given on a per-room basis may only be declared as type 'OTHER_ARRAY'.")
        #
        # Check that If a constructor initialized field has already been given a default value, 
        # it is only declares as OTHER_ARRAY.
        #
        #
        
        for k in list(SANTAH._vanilla_room_field_constructors.keys()):
            if k in self.field_contents:
                if self.field_type[k] != NamedTupleFieldType.OTHER_ARRAY:
                    raise Exception("Fields for which an explicit constructor is given on the proto-room level may only be declared as type 'OTHER_ARRAY'.")
        
        for k in list(tag_constructed_fields):
            if k in self.field_contents:
                if self.field_type[k] != NamedTupleFieldType.OTHER_ARRAY:
                    raise Exception("Fields for which an explicit constructor is given on the tag level may only be declared as type 'OTHER_ARRAY'.")
        
        
        
        # Get the values for all constructed fields
        #
        constructed_fields: List[str] = list(SANTAH._fields_constructor[self.underlyingTupleClass].keys())
        constructed_fields.extend(list(SANTAH._vanilla_room_field_constructors.keys()))
        constructed_fields.extend(list(tag_constructed_fields))
        init_field_content_underlying_tuple_class: Dict[str, Any] = copy.deepcopy(self.field_contents)
        init_field_content_vanilla_room: Dict[str, Any] = {}
        default_fs: List[str] = list(set(constructed_fields).difference(set(list(self.field_contents.keys()))))
        
        vanilla_room_fields: List[str] = [ e.value for e in SANTAH.vanilla_room_enum]
        for f in default_fs:
            init_field_content_underlying_tuple_class[f] = 0
            
        for c_f in constructed_fields:
            self.field_type[c_f] = NamedTupleFieldType.OTHER_ARRAY
            
        for f in init_field_content_underlying_tuple_class.keys():
            if f in vanilla_room_fields:
                init_field_content_vanilla_room[f] = init_field_content_underlying_tuple_class[f]
        
        content_dict: Dict[str, Any] = self.field_contents
        # Start at the proto room level:
        vanilla_room_constructed_: Dict[str, jArray] = {}
        # Do it this way, so that from that from the constructors point of view the field construction happens concurrently. 
        # This forbids interaction between constructed fields at the vanilla room level. 
        # I may change my mind about this in the future
        for f in list(SANTAH._vanilla_room_field_constructors.keys()):
            initted_vanilla_room = SANTAH.vanilla_room(**copy.deepcopy(init_field_content_vanilla_room))
            content = SANTAH._vanilla_room_field_constructors[f](initted_vanilla_room)
            vanilla_room_constructed_[f] = content
        for f in vanilla_room_constructed_.keys():
            content_dict[f] = vanilla_room_constructed_[f]
            init_field_content_underlying_tuple_class[f] = vanilla_room_constructed_[f]
            init_field_content_vanilla_room[f] = vanilla_room_constructed_[f]
        
        # Now initialize all the fields that have been decalared as requiring initialization at the TAG level.
        constructed_tag_fields: Dict[str, jArray] = {}
        
        for tag in implemented_tags:
            if tag in SANTAH.tag_based_room_constructor_fields:
                tag_field_constructors: Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]] = SANTAH.tag_based_room_constructor_fields[tag]
                for f_ in tag_field_constructors.keys():
                    
                    tag_nt: Type[NamedTuple] = SANTAH.my_tag_mapping[tag]
                    tag_constructor_args: Dict[str, Any] = {}
                    for f in tag_nt._fields:
                        tag_constructor_args[f] = init_field_content_underlying_tuple_class[f]
                    tag_ = tag_nt(**copy.deepcopy(tag_constructor_args))
                    vanilla_room = SANTAH.vanilla_room(**copy.deepcopy(init_field_content_vanilla_room))
                    field_content = tag_field_constructors[f_](vanilla_room, tag_)
                    constructed_tag_fields[f_] = field_content
        # Now move all the constructed tag fields into the content dict & make them available for initializing fields 
        # with constructors declared on the underlying tuple class level.
        for _c in constructed_tag_fields.keys():
            content_dict[_c] = constructed_tag_fields[_c]
            init_field_content_underlying_tuple_class[_c] = constructed_tag_fields[_c]
                    
            
        # Now construct the fields declared at the individual room level. This is a legacy feature and should not be used anyomore
        room_indiv_constructed_fields = list(SANTAH._fields_constructor[self.underlyingTupleClass].keys())
        for f in room_indiv_constructed_fields:
            cnt = SANTAH._fields_constructor[self.underlyingTupleClass][f](
                                self.underlyingTupleClass(**copy.deepcopy(init_field_content_underlying_tuple_class)))
            content_dict[f] = cnt

        # All fields in the room which need to be padded to roomsize to guarantee spatial consistency
        per_room_resize_fields: list[str]
        my_tags: list[Enum] = SANTAH.room_tags[self.underlyingTupleClass]
        per_room_resize_fields = list(itertools.chain.from_iterable([list(SANTAH.roomsized_tag_fields[e]) for e in my_tags if e in SANTAH.roomsized_tag_fields]))
        per_room_resize_fields += [e.value for e in SANTAH.roomsized_room_fields]
        if SANTAH.display_height is None or SANTAH.display_width is None:
            raise Exception(
                "ROOM_SIZED_ARRAY fields are present but display_height/display_width were not "
                "registered. Pass display_height and display_width to SANTAH.register_proto_room.")
        vertical_offset_val = int(content_dict[RequiredRoomFields.VERTICAL_OFFSET.value][0])
        for f in content_dict.keys():
            if f not in per_room_resize_fields:
                continue
            print(f)
            field_arr = content_dict[f]
            # Build the full-display-size shape: same as field_arr but dim 1 -> display_height
            full_shape = list(field_arr.shape)
            full_shape[1] = SANTAH.display_height
            if full_shape == field_arr.shape:
                continue
            padded = jnp.zeros(full_shape, dtype=field_arr.dtype)
            # start_indices: 0 for every dim except dim 1 which is vertical_offset
            start_indices = [0] * len(field_arr.shape)
            start_indices[1] = vertical_offset_val
            padded = jax.lax.dynamic_update_slice(padded, field_arr, start_indices=tuple(start_indices))
            content_dict[f] = padded

        def jittable_initialisation(content: Dict[str, Any], tuple_class: NamedTuple):
            return tuple_class(**content)
        return jax.jit(partial(jittable_initialisation, content=content_dict, tuple_class=self.underlyingTupleClass))
    
    def get_serialisation_function(self, generate_writer: bool = True) -> Tuple[Callable[[NamedTuple], jArray], Callable[[NamedTuple, jArray], NamedTuple], int]:
        """Generates a function that takes in a NamedTuple representing a room 
           and returns a serialised array version of the persistant fields in the named tuple, 
           i.e. the fields that need to be stored to the global store. 
           Also generates the corresponding deserialisation function which takes in an initialized NamedTuple and 
           the persistence storage associated with this room and loads the values from the persistence storage into the 
           named_tuple
        Returns:
            Tuple[Callable[[NamedTuple], jArray], int]: Returns a serialisation function and and integer that gives 
                the size of all persisting fields.
        """
        # Split all serializable fields according to their type.
        # Also collect partial + full serialisation functions 
        # for all named-tuple fields.
        size: int = 0
        singleton_integer_fields: List[str] = []
        named_tuple_fields: List[str] = []
        named_tuple_partial_deserialise: List[Callable[[NamedTuple, jArray], jArray]] = []
        named_tuple_partial_serialisation: List[Callable[[NamedTuple, jArray], jArray]] = []
        named_tuple_full_serialisation: List[Callable[[NamedTuple], jArray]] = []
        named_tuple_full_deserialise: List[Callable[[NamedTuple, jArray], jArray]] = []
        named_tuple_sizes: List[int] = []
        named_tuple_stack_heights: List[int] = []
        for f in self.present_fields:
            if self.field_type[f] in (NamedTupleFieldType.OTHER_ARRAY, NamedTupleFieldType.ROOM_SIZED_ARRAY):
                continue
            elif self.field_type[f] == NamedTupleFieldType.INTEGER_SCALAR:
                if not self.field_persistent[f]:
                    continue
                else:
                    size += 1
                    singleton_integer_fields.append(f)
            elif self.field_type[f] == NamedTupleFieldType.NAMED_TUPLE_STACK:
                named_tuple_fields.append(f)
                named_tuple_partial_deserialise.append(
                    SANTAH.partial_deserialisations[self.field_named_tuples[f]])
                named_tuple_partial_serialisation.append(
                    SANTAH.partial_serialisations[self.field_named_tuples[f]])
                named_tuple_full_serialisation.append(
                    SANTAH.full_serialisations[self.field_named_tuples[f]])
                named_tuple_full_deserialise.append(
                    SANTAH.full_deserializations[self.field_named_tuples[f]])
                named_tuple_stack_heights.append(len(self.field_contents[f]))
                
                tup_size: int = SANTAH.partially_serialised_sizes[self.field_named_tuples[f]]
                named_tuple_sizes.append(tup_size)
                size += tup_size*len(self.field_contents[f])
                
                
        # The actual function used to write a room to persistence. 
        # This function essentially picks out all the attributes of the room that 
        # were declared to be serializable & concatenates them into a long 
        # array which can be written to global storage.
        def _write_self_to_persistence(_tuple: NamedTuple, _int_fields: List[str], _named_tup_fields: List[str], 
                                       _named_tup_deserialisation_function: List[Callable[[jArray], NamedTuple]], 
                                       _named_tup_partial_serialisation_fun: List[Callable[[NamedTuple], jArray]], 
                                       _named_tuple_sizes: List[int], 
                                       _named_tuple_stack_heights: List[int], 
                                       _full_size: int
                                       ):
            full_serialised_array: jArray = jnp.zeros((_full_size, ), dtype=jnp.int32)
            current_offset = 0
            for i, d in enumerate(_int_fields):
                full_serialised_array = full_serialised_array.at[i].set(getattr(_tuple, d)[0])
                current_offset += 1
            for d in zip(_named_tup_fields, 
                         _named_tup_deserialisation_function, 
                         _named_tup_partial_serialisation_fun, 
                         _named_tuple_stack_heights, 
                         _named_tuple_sizes):
                _field, _deserialise, _serialise_part, _height, _size = d
                for i in range(_height):
                    current_arr: jArray = getattr(_tuple, _field)[i, ...]
                    nt_repr: NamedTuple = _deserialise(current_arr)
                    persistence_arr: jArray = _serialise_part(nt_repr)
                    full_serialised_array = full_serialised_array.at[current_offset:current_offset+_size].set(persistence_arr)
                    current_offset += _size
            return full_serialised_array
        
        
        # The Persistence load function.
        # receives a slice of the global storage corresponding to the serialisable fields of this room
        # & reconstructs the room-named tuple from the serialised fields & default values for all other fields.
        #
        #
        def _load_self_from_persistence(_tuple: NamedTuple, _persistence_storage: jArray, _int_fields: List[str], 
                                        _named_tup_fields: List[str], 
                                       _named_tup_serialisation_function: List[Callable[[NamedTuple], jArray]], 
                                       _named_tup_partial_deserialisation_function: List[Callable[[NamedTuple], jArray]],
                                       _named_tup_full_deserialisation_function: List[Callable[[jArray], NamedTuple]], 
                                       _named_tuple_sizes: List[int], 
                                       _named_tuple_stack_heights: List[int], 
                                       _all_fields: List[str], 
                                       _tuple_class: NamedTuple
                                       ):
            field_contents: Dict[str, jArray] = {}
            for f in _all_fields:
                field_contents[f] = getattr(_tuple, f)
            
            current_offset = 0
            for i, d in enumerate(_int_fields):
                field_contents[d] = jnp.array([_persistence_storage[i]])
                current_offset += 1
            for d in zip(_named_tup_fields, 
                         _named_tup_serialisation_function, 
                         _named_tup_partial_deserialisation_function, 
                         _named_tup_full_deserialisation_function, 
                         _named_tuple_sizes, 
                         _named_tuple_stack_heights):
                _field, _serialise_fully, _deserialize_part, _deserializy_fully, _size, _height = d
                tuple_stack: jArray = field_contents[_field]
                for i in range(_height):
                    tmp = tuple_stack[i, ...]
                    curr_tup: NamedTuple = _deserializy_fully(tmp)
                    _arr_persist = _persistence_storage[current_offset: current_offset+_size]
                    curr_tup = _deserialize_part(curr_tup, _arr_persist)
                    serialised_tup: jArray = _serialise_fully(curr_tup)
                    tuple_stack = tuple_stack.at[i, ...].set(serialised_tup)
                    current_offset += _size
                field_contents[_field] = tuple_stack
            ret = _tuple_class(**field_contents)
            return ret
        
        if generate_writer:
            wrapped_persistence_writer: Callable[[NamedTuple], jArray] = partial(
                _write_self_to_persistence, 
                    _int_fields = singleton_integer_fields, _named_tup_fields = named_tuple_fields, 
                    _named_tup_deserialisation_function = named_tuple_full_deserialise, 
                    _named_tup_partial_serialisation_fun = named_tuple_partial_serialisation, 
                    _named_tuple_sizes = named_tuple_sizes, 
                    _named_tuple_stack_heights = named_tuple_stack_heights, 
                    _full_size = size
            )
            wrapped_persistence_writer = jax.jit(wrapped_persistence_writer)
        
        jitted_persistence_loader: Callable[[NamedTuple, jArray], NamedTuple] = partial(
            _load_self_from_persistence, 
            _int_fields = singleton_integer_fields,
            _named_tup_fields = named_tuple_fields, 
            _named_tup_serialisation_function = named_tuple_full_serialisation, 
            _named_tup_partial_deserialisation_function = named_tuple_partial_deserialise,
            _named_tup_full_deserialisation_function = named_tuple_full_deserialise, 
            _named_tuple_sizes = named_tuple_sizes, 
            _named_tuple_stack_heights = named_tuple_stack_heights, 
            _all_fields = self.present_fields, 
            _tuple_class = self.underlyingTupleClass
        )
        jitted_persistence_loader = jax.jit(jitted_persistence_loader)
        
        if generate_writer:
            return wrapped_persistence_writer, jitted_persistence_loader, size
        else:
            return jitted_persistence_loader, size
        
    def connect_to(self, my_location: RoomConnectionDirections, other_room: Room, other_location: RoomConnectionDirections):
        # 
        # Generate the graph structure which describes the room layout.
        #
        if my_location == RoomConnectionDirections.LEFT:
            self.connection_object.left = other_room.connection_object
            
        if my_location == RoomConnectionDirections.RIGHT:
            self.connection_object.right = other_room.connection_object
        if my_location == RoomConnectionDirections.UP:
            self.connection_object.up = other_room.connection_object
        if my_location == RoomConnectionDirections.DOWN:
            self.connection_object.down = other_room.connection_object
        
        # Create symmetric connection
        if other_location == RoomConnectionDirections.UP:
            other_room.connection_object.up = self.connection_object
        
        if other_location == RoomConnectionDirections.DOWN:
            other_room.connection_object.down = self.connection_object
            
        if other_location == RoomConnectionDirections.LEFT:
            other_room.connection_object.left = self.connection_object
        if other_location == RoomConnectionDirections.RIGHT:
            other_room.connection_object.right = self.connection_object    
        
    
class NewPyramidLayout:
    """A fresh pyramid layout in a vague attempt to absolve myself from the sins of my bad design decisions. 
        They still haunt me to this day.
    """
    def __init__(self):
        self.running_counter: int = 0
        self.rooms: Dict[int, Room] = {}
        self.persistence_writer: Callable[[NamedTuple, jArray], jArray] = None
        self.persistence_loader: Callable[[NamedTuple, jArray], NamedTuple] = None
        self.raise_proto_room_to_specific: Dict[int, Callable[[NamedTuple, NamedTuple], NamedTuple]]= {}
        self.lower_specific_room_to_proto_room: Dict[int, Callable[[NamedTuple], NamedTuple]] = {}       
        self.initial_persistence_storage: jArray
        
        # This is ~~ mostly ~~ just used for stashing writers and loaders so that 
        # we don't have to recompute them when we generate wrapper functions that handle room-specific functionality
        self.room_persistence_writers: Dict[int, Callable[[NamedTuple], jArray]] = {}
        self.room_persistence_loaders: Dict[int, Callable[[NamedTuple, jArray], NamedTuple]] = {}
        self.static_jitted_raising_function: Callable[[NamedTuple, jArray, int], NamedTuple] = None
        self.single_room_proto_loader: Dict[int, Callable[[jArray, jArray], NamedTuple]] = {}
        self.single_room_proto_writer: Dict[int, Callable[[jArray, NamedTuple, jArray], jArray]] = {}
    

class MontezumaRoomLayoutAuthority():
    # List of all enrolled named tuples.
    _enrolled_named_tuple: list[type[NamedTuple]] = []
    # All named tuples that have been enrolled as Tag
    _enrolled_room_tags: list[type[NamedTuple]] = []
    # The one named tuple that has been enrolled as the Room.
    _room_named_tuple: Optional[NamedTuple] = None
    
    # This is used during the actual Training. Maps typename of named tuple 
    # To specifications of annotations.
    _type_name_to_annos: dict[str, dict[str, list[object]]] = {}
    
    # Only used during construction
    _type_to_annos: dict[type[NamedTuple], dict[str, list[object]]] = {}
    
    
    # Infrastructure inherited from SANTAH
   
    tag_based_room_constructor_fields: Dict[type[NamedTuple], Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]] = None
    #
    # TODO: most of this shit should really only be called once during level creation. Actual gameplay infra should be able 
    # to be run from
    #
    #
    
    @classmethod
    def register_constructed_fields(cls, constructed_fields: Dict[type[NamedTuple], Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]] = {}):
        """Gives the option to register a set of room tags. 
            Room tags are used to signify, that a room has certain features, such as ladders/ LAZERWALLS/ or a Bottomless pit. 
            Room tags are always paired with a NamedTuple representing the fields which are needed for the 
                functionality described by the TAG. 
            In the backend, the necessary Infrastructure is generated to allow for wrapped handling of the 
            room behavior solely in terms of the Proto room and all supported Tags

        Args:
            room_tags (Enum): An Enum of all Tags supported by the Montezuma Implementation. 
                Values in the Enum are Named tuples specifying the fields required to support the tag
            constructed_fields:  Dict[Enum, Dict[str, Callable[[NamedTuple], jArray]]] = {} : Enables support for constructed 
                fields at the tag level. 
                The Dict needs to map Keys from the room_tags enum to dictionaries mapping {field_name -> constructors}.
                The constructors receive both a Vanilla Room representing the already set fields (all fields defined via 
                the room API in game & the constructed fields from the Vanilla Room) and the appropriate room tag, 
                with all fields defined via the Room-API and 0 default values for the not already constructed fields. 
                

        """
        
        for e in constructed_fields:
            if not e in cls._enrolled_room_tags:
                raise Exception(f"Room Tag {e.__name__} needs to be registered as a room tag first.")
            
            
        # Check if all constructed fields are actually present in their respective tags.
        existing_room_tags = set([e for e in cls._enrolled_room_tags])
        t_constrs = set(list(constructed_fields.keys()))
        if not t_constrs.issubset(existing_room_tags):
            raise Exception(f"The keys of the 'constructed_fields' needs to be a subset of the fields in the respecitv tags.")
        
        # Check if fields for which constructors are defined are acually present in the respective tags.
        for k in constructed_fields.keys():
            tag_nt: Type[NamedTuple] = k
            if not set(constructed_fields[k].keys()).issubset(set(tag_nt._fields)):
                raise Exception(f"For Room-Tag {tag_nt.__name__}: User attempted to register a constructed field which is not part of the tag.")
        cls.tag_based_room_constructor_fields = constructed_fields
        
                
    
    
    
    
    @classmethod
    def valide_field_contens(cls, tag: Optional[type[NamedTuple]], field: str, content: jnp.ndarray|list[NamedTuple]) -> bool:
        """Validates the prospective contents of a named-tuple field.

        Args:
            tag (Optional[type[NamedTuple]]): _description_
            field (str): _description_
            content (jnp.ndarray | list[NamedTuple]): _description_

        Raises:
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_

        Returns:
            bool: _description_
        """
    
    
    @classmethod
    def is_namedtuple_class(cls, nt_class) -> bool:
        # Hacky, attempts to check whether a type is a NamedTuples.
        # Can't do that via is_subtype as NamedTuples are not really types...
        return (
            isinstance(nt_class, type)
            and issubclass(nt_class, tuple)
            and hasattr(nt_class, "_fields")
            and hasattr(nt_class, "_asdict")
            and isinstance(getattr(nt_class, "_fields"), tuple)
        )
    
    def _fetch_custom_annotations_per_field(tup: type[NamedTuple]) -> dict[str, list[object]]:
        ret_annos: dict[str, list[object]] = {}
        hints = get_type_hints(tup, include_extras=True)
        tup_name: str = tup.__name__
        fields: list[str] = list(hints.keys())
        if len(field) > len(set(field)):
            raise Exception(f"NamedTuple class {tup.__name__} has multiple fields with the same name. This is forbidden.")
        for field, f_hint in hints.items():
            if get_origin(f_hint) is not Annotated:
                raise Exception(f"Field {field} of NamedTuple {tup_name} is not annotated via Annotation[].")
            base_type, custom_annos = get_args(f_hint)
            ret_annos[field] = list(custom_annos)
        return ret_annos
    
    @classmethod
    def get_annotations(cls, tup: type[NamedTuple], field: str) -> list[object]:
        if not tup in cls._type_to_annos:
            raise Exception(f"Attempted to retreive annotations for non-registered Named Tuple {tup.__name__}")
        annos: dict[str, list[object]] = cls._fetch_custom_annotations_per_field(tup)
        if field not in annos:
            raise Exception(f"Attempted to retreive annotations")
        return annos[field]
    
    @classmethod
    def _check_named_tuple_annotations(cls, tup: type[NamedTuple]) -> None:
        hints = get_type_hints(tup, include_extras=True)
        tup_name: str = tup.__name__
        fields: list[str] = list(hints.keys())
        if len(field) > len(set(field)):
            raise Exception(f"NamedTuple class {tup.__name__} has multiple fields with the same name. This is forbidden.")
        for field, f_hint in hints.items():
            if get_origin(f_hint) is not Annotated:
                raise Exception(f"Field {field} of NamedTuple {tup_name} is not annotated via Annotation[].")
            base_type, custom_annos = get_args(f_hint)
            custom_annos = tuple(custom_annos)
            if STATIC not in custom_annos and DYNAMIC not in custom_annos:
                raise Exception(f"NamedTuple {tup_name}, field {field}:: All fields are required to have either a STATIC or DYNAMIC annotation,")
            if STATIC in custom_annos and DYNAMIC in custom_annos:
                raise Exception(f"NamedTuple {tup_name}, field {field}:: Is both annotated with STATIC and DYNAMIC; mutually exclusive")
            # Check static annotations for validity:
            if DYNAMIC in custom_annos:
                if ROOM_SHAPED in custom_annos or CONSTANT_SHAPE in custom_annos:
                    raise Exception(f"NamedTuple {tup_name}, field {field}:: Dynamic fields may only be SINGLETON_INTEGER or NAMED_TUPLE_STACK")
                if SINGLETON_INT in custom_annos and NAMED_TUPLE_STACK in custom_annos:
                    raise Exception(f"NamedTuple {tup_name}, field {field}:: Dynamic fields may not be tagged as both SINGLETON_INT and NAMED_TUPLE_STACK")
                if SINGLETON_INT not in custom_annos and NAMED_TUPLE_STACK not in custom_annos:
                    raise Exception(f"NamedTuple {tup_name}, field {field}:: Dynamic fields need to be either annotated as SINGLETON_INT or NAMED_TUPLE_STACK")
                if NAMED_TUPLE_STACK in custom_annos:
                    nt_type_annotations = [c for c in custom_annos if cls.is_namedtuple_class(c)]
                    if len(nt_type_annotations) == 0:
                        raise Exception(f"NamedTuple {tup_name}, field {field}::  If a field is annotated as NAMED_TUPLE_STACK, an additional NamedTuple class needs to be provided in the annotations which specifies the type of the stacked named tuple.")
                    if len(nt_type_annotations > 1):
                        raise Exception(f"NamedTuple {tup_name}, field {field}::  If a field is annotated as NAMED_TUPLE_STACK, an additional NamedTuple class needs to be provided in the annotations which specifies the type of the stacked named tuple.")
                return None
            if STATIC in custom_annos:
                allowed = set([ROOM_SHAPED, NAMED_TUPLE_STACK, SINGLETON_INT, CONSTANT_SHAPE])
                if len(set(custom_annos).intersection(allowed)) < 1:
                    raise Exception(f"NamedTuple {tup_name}, field {field}:: All fields need to be annotated with one of the following: [ROOM_SHAPED, NAMED_TUPLE_STACK, SINGLETON_INT, CONSTANT_SHAPE]")
                if len(set(custom_annos).intersection(allowed)) < 1:
                    raise Exception(f"NamedTuple {tup_name}, field {field}:: All static fields must have exactly one of the following annotations: [ROOM_SHAPED, NAMED_TUPLE_STACK, SINGLETON_INT, CONSTANT_SHAPE]")
                if NAMED_TUPLE_STACK in custom_annos:
                    nt_type_annotations = [c for c in custom_annos if cls.is_namedtuple_class(c)]
                    if len(nt_type_annotations) == 0:
                        raise Exception(f"NamedTuple {tup_name}, field {field}::  If a field is annotated as NAMED_TUPLE_STACK, an additional NamedTuple class needs to be provided in the annotations which specifies the type of the stacked named tuple.")
                    if len(nt_type_annotations > 1):
                        raise Exception(f"NamedTuple {tup_name}, field {field}::  If a field is annotated as NAMED_TUPLE_STACK, an additional NamedTuple class needs to be provided in the annotations which specifies the type of the stacked named tuple.")
                
            return None
    
    @classmethod
    def register_named_tuple_class(cls, tup: type[NamedTuple], 
            register_as_tag: bool = False, 
            register_as_room: bool = False)->None:
        if register_as_room and register_as_tag:
            raise Exception(f"Cannot register NamedTuple {tup.__name__} as both Tag and Room.")
        cls._enrolled_named_tuple.append(tup)
        if register_as_tag:
            cls._enrolled_room_tags.append(tup)
        if register_as_room:
            cls._room_named_tuple = tup
        annos: dict[str, list[object]] = cls._fetch_custom_annotations_per_field(tup)
        cls._type_name_to_annos[tup.__name__] = annos
        cls._type_to_annos[tup] = annos
    
    @classmethod
    def _get_anno_tuple_stack_type(cls, annos: list[object]) -> type[NamedTuple]|None:
        if not NAMED_TUPLE_STACK in annos:
            return None
        nt_type_annotations = [c for c in annos if cls.is_namedtuple_class(c)]
        return nt_type_annotations[0]
    
    @classmethod
    def _check_eligible_named_tuple_stack_type(cls, stack_type: type[NamedTuple]) -> bool:
        if stack_type not in cls._enrolled_named_tuple:
            return False
        if stack_type in cls._enrolled_room_tags or stack_type == cls._room_named_tuple:
            return False
        annos: dict[str, list[object]] = cls._type_to_annos[stack_type]
        
    
    @classmethod
    def _validate_registered_named_tuples(self) -> None:
        """
        Named Tuple classes that are registered as the Content of a NamedTupleStack may only contain scalar fields (static and dynamic).
        Also, they may not be registred as either room or tag themselves.
        """
        
        
        
    @classmethod
    def build_layout(cls, layout: PyramidLayout, storage_prefix: str = "mzuma_layout") -> None:
        for room in layout.rooms.values():
            if not type(room) in cls._room
        