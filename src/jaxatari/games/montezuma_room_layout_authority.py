from jaxatari.games.jax_mzuma_utils import SANTAH, DYNAMIC, STATIC, CONSTANT_SHAPE, ROOM_SHAPED, NAMED_TUPLE_STACK, SINGLETON_INT, TagNamedTuple, VanillaRoom, RoomConnectionObject, RequiredRoomFields, NamedTupleFieldType, RoomConnectionDirections, Room
from typing import NamedTuple, get_type_hints, Annotated, get_origin, get_args, Optional, Type, Callable, Dict, Any, List, Tuple
from enum import Enum
import warnings
from jaxatari.games.jax_montezuma_constants import *
from jax import Array as jArray
import jax.numpy as jnp
from jaxatari.games.jax_mzuma_utils import PyramidLayout


class NewRoom:
    def __init__(self, room_id: int, base_class: type[NamedTuple], tags: list[type[NamedTuple]] = []):
        #
        # TODO: Have this called by the new room authority class. 
        # don't let user touch it themselves
        #
        #
        self.base_class: type[NamedTuple] = base_class
        self.room_id: int = room_id
        self.tags = tags
        self.connection_object: RoomConnectionObject = RoomConnectionObject(room_id=self.room_id)
        self.base_class_fields: Dict[str, jArray|List[NamedTuple]] = {}
        self.base_class_fields[RequiredRoomFields.ROOM_ID.value] = jnp.array([self.room_id], dtype=jnp.uint16)
        self.tag_fields: dict[type[NamedTuple], dict[str, jArray|List[NamedTuple]]] = {}
         
        self.base_class_required_fields: List[str] = list(self.base_class._fields)
        if not RequiredRoomFields.ROOM_ID.value in self.base_class_fields:
            raise Exception("All NamedTuples representing individual rooms are required to have a 'ROOM_ID' field")
        
        
        
    def set_field(self, field_name: str, content: jnp.ndarray|List[NamedTuple], tag: type[NamedTuple]|None):
        """The main method through which fields of individual rooms are set.
        """
        if tag is not None and tag not in self.tags:
            raise Exception(f"Attemptet to set field for tag {tag.__name__}; Tag not present in the room")
        # Check whether valid field content was passed.
        if field_name == RequiredRoomFields.ROOM_ID.value:
            raise Exception("Cannot overwrite the fixed field 'ROOM_ID'")

        if not MontezumaRoomLayoutAuthority.validate_field_contents(tag, field_name, content):
            tag_name = tag.__name__ if tag else self.base_class.__name__
            raise Exception(f"Validation failed for field '{field_name}' in {tag_name}")

        # Store content
        if tag is None:
            self.base_class_fields[field_name] = content
        else:
            if tag not in self.tag_fields:
                self.tag_fields[tag] = {}
            self.tag_fields[tag][field_name] = content
        
    def _build_fields(self) -> None:
        # Validate required base-room fields.
        required_base_fields = MontezumaRoomLayoutAuthority.get_required_fields(None)
        missing_base_fields = [f for f in required_base_fields if f not in self.base_class_fields]
        if len(missing_base_fields) > 0:
            raise Exception(
                f"Room with ID {self.room_id} is missing required base-room fields: {missing_base_fields}"
            )

        # Check that all tags that have fields are actually present.
        needed_tags = set(self.tags)
        # Tags without required fields (e.g. only constructed fields) can be represented by an empty dict.
        flag_tags = [t for t in self.tags if len(MontezumaRoomLayoutAuthority.get_required_fields(t)) == 0]
        for t in flag_tags:
            if t not in self.tag_fields:
                self.tag_fields[t] = {}

        # Validate that all required fields for present tags have been set.
        for t in self.tags:
            required_tag_fields = MontezumaRoomLayoutAuthority.get_required_fields(t)
            if t not in self.tag_fields:
                continue
            missing_tag_fields = [f for f in required_tag_fields if f not in self.tag_fields[t]]
            if len(missing_tag_fields) > 0:
                raise Exception(
                    f"Room with ID {self.room_id} is missing required fields for tag {t.__name__}: {missing_tag_fields}"
                )

        present_tags = set(list(self.tag_fields.keys()))
        if len(present_tags.intersection(needed_tags)) != len(needed_tags):
            superf_tags = list(present_tags.difference(needed_tags))
            if len(superf_tags) > 0:
                raise Exception(f"Room with ID {self.room_id} has tags it is not supposed to have: {superf_tags}")
            missing_tags = list(needed_tags.difference(present_tags))
            if len(missing_tags) > 0:
                raise Exception(f"Room with ID {self.room_id} is missing declared room tags: {missing_tags}")

        # Construct fields concurrently: build instances with all fields present (unconstructed + empty arrays for constructed),
        # then call all constructors in a single pass so no constructed fields are visible yet.
        
        # Build base room instance with empty arrays for its constructed fields
        base_room_fields = dict(self.base_class_fields)
        base_room_constructors = MontezumaRoomLayoutAuthority.get_base_constructors()
        for field_name in base_room_constructors:
            base_room_fields[field_name] = jnp.array([])
        base_room_instance = self.base_class(**base_room_fields)
        
        # Build tag instances with empty arrays for their constructed fields
        tag_instances: Dict[type[NamedTuple], NamedTuple] = {}
        for t in self.tags:
            tag_fields = dict(self.tag_fields[t]) if t in self.tag_fields else {}
            tag_constructors = MontezumaRoomLayoutAuthority.get_tag_constructors(t)
            for field_name in tag_constructors:
                tag_fields[field_name] = jnp.array([])
            tag_instances[t] = t(**tag_fields)
        
        # Call all base room constructors and store results
        for field_name, constructor in base_room_constructors.items():
            # For base room constructors, pass the first available tag instance if any tags exist
            tag_param = None
            content = constructor(base_room_instance, tag_param)
            if not MontezumaRoomLayoutAuthority.validate_field_contents(tag=None, field=field_name, 
                                                                    content=content):
                raise Exception(f"Constructor for field {field_name} of base-room returned value of invalid type.")
            self.base_class_fields[field_name] = content
        
        # Call all tag constructors and store results
        for t in self.tags:
            tag_constructors = MontezumaRoomLayoutAuthority.get_tag_constructors(t)
            for field_name, constructor in tag_constructors.items():
                content = constructor(base_room_instance, tag_instances[t])
                if not MontezumaRoomLayoutAuthority.validate_field_contents(tag=t, field=field_name, 
                                                                            content=content):
                    raise Exception(f"Constructor for field {field_name} in room-tag {t.__name__} returned value of invalid type.")
                self.tag_fields[t][field_name] = content
        
            
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
        
    def create_new_room(self, tags: Tuple[Type[Enum]] = ()):
        """The main function used to construct new rooms. 
            Automatically generates all persistence infrastructure required by the framework.

        Args:
            tags (Tuple[Type[Enum]], optional): Tags representing the functionality 
                which this room should be implemented. A room can implement arbitrary many tags, 
                all infrastructure that is required to support this is generated automatically on game-startup.. Defaults to ().

        """
        if not isinstance(tags, Tuple):
            raise Exception("Tags need to be specified as a tuple.")
        if MontezumaRoomLayoutAuthority._room_named_tuple is None:
            raise Exception(f"No base room named tuple registered.")
        if not MontezumaRoomLayoutAuthority.check_tag_overlap(tags):
            raise Exception(f"The given set of tags overlaps.")
        new_room = NewRoom(room_id=self.running_counter, base_class=MontezumaRoomLayoutAuthority._room_named_tuple, 
                           tags=tags)
        self.running_counter += 1
        self.rooms[new_room.room_id] = new_room
        # Generate lowering & raising functions necessary for the function wrappers.
        return new_room
    
class RoomPersistenceStorage():
    # In all, room dimension is along the first dimension
    # Contents for all static fields in the base room. Key is field name, contains stacked contents (along the first axis) of all static fields.
    base_room_static_fields: dict[str, jArray] = {}
    # Same for the dynamic fields in the base room.
    base_room_dynamic_fields: dict[str, jArray] = {}
    # Content for tag fields are exactly the same, but outer level is another dictionary 
    # mapping tag_name to the field dict.
    tag_static_fields: dict[str, dict[str, jArray]] = {}
    tag_dynamic_fields: dict[str, dict[str, jArray]] = {}
    # Maps name of the base room field to the slice size occupied by the per-room contents along the first 
    # axis. As jax requires input shapes to stay static, if the field contains a stack of something 
    # (right now only named tuples), the stack size gets padded for each room/ named tuple to the biggest 
    # occuring stack-size among rooms/ tags of that type. 
    base_room_field_slice_size: dict[str, int] = {}
    # Same as above, but top level dict maps tag_name.
    tag_field_slice_size: dict[str, dict[str, int]] = {}
    # Maps field -> (room_id -> offset). Offset at which the field-contents for the given room can be found
    room_field_id_offset: dict[str, dict[int, int]] = {}
    # Maps tag_name -> (field -> (room_id -> offset)), otherwise same as above.
    tag_field_room_id_offset: dict[str, dict[str, dict[int, int]]] = {}
    # Included padding along the first axis for the field contents of each room. 
    # This only really matters for named_tuple_stack_fields, all other fields have static shape for all rooms anyway. 
    # Maps (field_name -> (room_id -> padding)) (padding is 0 if it's not a named-tuple field)
    base_room_ntstack_padding: dict[str, dict[int, int]] = {}
    # Same as above but maps tag_name -> (field_name -> (room_id -> padding))
    tag_field_ntstack_padding: dict[str, dict[str, dict[int, int]]] = {}

class MontezumaRoomLayoutAuthority():
    # List of all enrolled named tuples.
    _enrolled_named_tuple: list[type[NamedTuple]] = []
    # All named tuples that have been enrolled as Tag
    _enrolled_room_tags: list[type[NamedTuple]] = []
    # The one named tuple that has been enrolled as the Room.
    _room_named_tuple: Optional[type[NamedTuple]] = None
    
    # This is used during the actual Training. Maps typename of named tuple 
    # To specifications of annotations.
    _type_name_to_annos: dict[str, dict[str, list[object]]] = {}
    
    # Only used during construction
    _type_to_annos: dict[type[NamedTuple], dict[str, list[object]]] = {}
    
    
    # Infrastructure inherited from SANTAH
   
    tag_constructor_fields: Dict[type[NamedTuple], Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]] = {}
    base_room_constructed_fields: Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]] = None
    #
    # TODO: most of this shit should really only be called once during level creation. Actual gameplay infra should be able 
    # to be run from
    #
    #
    
    @classmethod
    def register_constructors(cls, constructors: Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]], tag: type[NamedTuple]|None = None) -> None:
        if tag is None:
            if cls._room_named_tuple is None:
                raise Exception("Cannot set constructed fields for base room without registering Base Room class first.")
            if cls.base_room_constructed_fields is not None:
                raise Exception("Constructed fields for base room type was already set.")
            avail_base_fields: list[str] = cls._fields_for_nt(cls._room_named_tuple)
            for field, constr in constructors.items():
                if field not in avail_base_fields:
                    raise Exception(f"Attempted to register constructor for non-existent field {field} of Base Room Class")

            cls.base_room_constructed_fields = constructors
            return None
        
        if tag not in cls._enrolled_room_tags:
            raise Exception(f"Attempted to register constructors for Named Tuple Class {tag.__name__} - needs to be registered as room tag.")
        
        if tag in cls.tag_constructor_fields:
            raise Exception(f" Constructors for room tag {tag.__name__} were already registered.")
        avail_tag_fields: list[str] = cls._fields_for_nt(tag)
        for field, constr in constructors.items():
            if field not in avail_tag_fields:
                raise Exception(f"Attempted to register constructor for field l{field} of room-tag {tag.__name__} - field does not exist")
        cls.tag_constructor_fields[tag] = constructors
        
    @classmethod
    def check_tag_overlap(cls, tags: Tuple[type[NamedTuple]]) -> bool:
        """
        Checks whether fields of the given set of tags overlap.
        """
        seen_fields: set[str] = set()
        for tag in tags:
            tag_fields = set(cls._fields_for_nt(tag))
            if len(seen_fields.intersection(tag_fields)) > 0:
                return False
            seen_fields.update(tag_fields)
        return True
        
    @classmethod
    def get_required_fields(self, tag: type[NamedTuple]|None) -> list[str]:
        """Returns the list of explicitly required fields for the given room tag or the 
            base room-type.

        Args:
            tag (type[NamedTuple] | None): _description_

        Returns:
            list[str]: _description_
        """
        if tag is None:
            if self._room_named_tuple is None:
                raise Exception("No room named tuple has been registered yet.")
            base_fields: list[str] = self._fields_for_nt(self._room_named_tuple)
            if self.base_room_constructed_fields is None:
                return base_fields
            constructed_fields = set(self.base_room_constructed_fields.keys())
            return [field for field in base_fields if field not in constructed_fields]

        if tag not in self._enrolled_room_tags:
            raise Exception(f"Attempted to retrieve required fields for non-registered room tag {tag.__name__}")

        tag_fields: list[str] = self._fields_for_nt(tag)
        if tag not in self.tag_constructor_fields:
            return tag_fields

        constructed_fields = set(self.tag_constructor_fields[tag].keys())
        return [field for field in tag_fields if field not in constructed_fields]

    
    @classmethod
    def get_base_constructors(cls) -> Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]:
        if cls.base_room_constructed_fields is None:
            warnings.warn(f"No constructed fields registered for base-room class {cls._room_named_tuple.__name__}.")
            return {}
        return cls.base_room_constructed_fields
    
    @classmethod
    def get_tag_constructors(cls, tag: type[NamedTuple]) -> Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]:
        if not tag in cls.tag_constructor_fields:
            warnings.warn(f"Attempted to retreive constructors for room tag {tag.__name__} - no constructors found.")
            return {}
        return cls.tag_constructor_fields[tag]
    
    @classmethod
    def is_room_tag(cls, nt_type: type[NamedTuple]) -> bool:
        return nt_type in cls._enrolled_room_tags
    
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
        cls.tag_constructor_fields = constructed_fields
        
                
    
    
    
    
    @classmethod
    def validate_field_contents(cls, tag: Optional[type[NamedTuple]], field: str, content: jnp.ndarray|list[NamedTuple]) -> bool:
        """Validates the prospective contents of a named-tuple field against the annotations given in the corresponding NamedTuple.

        Args:
            tag (Optional[type[NamedTuple]]): RoomTag which the field belongs to. If no tag is given, 
                field belongs to the underlying basic room class.
            field (str): Name of the field.
            content (jnp.ndarray | list[NamedTuple]): Contents of the field. Either a numpy array or if the field contains 
                a NamedTupleStack a list of named tuples of the same (already registered) type

        Returns:
            bool: _description_
        """
        target_tup = tag if tag is not None else cls._room_named_tuple
        if target_tup is None:
            raise Exception("No room named tuple has been registered yet.")
        
        if target_tup not in cls._type_to_annos:
            raise Exception(f"NamedTuple {target_tup.__name__} has not been registered.")
        
        annos = cls._type_to_annos[target_tup]
        if field not in annos:
            raise Exception(f"Field {field} is not present in NamedTuple {target_tup.__name__}")
        
        field_annos = annos[field]
        
        # Check if it's a NamedTupleStack
        if NAMED_TUPLE_STACK in field_annos:
            if not isinstance(content, list):
                return False
            stack_type = cls._get_anno_tuple_stack_type(field_annos)
            if stack_type is None:
                return False
            if stack_type not in cls._enrolled_named_tuple:
                raise Exception(f"Cannot accept non-registered tuple-type {stack_type.__name__} as type for field {field}")
            if stack_type in cls._enrolled_room_tags or stack_type == cls._room_named_tuple:
                raise Exception(f"Cannot accept named-tuple type {stack_type.__name__} as type for field {field}; Only types not registered as RoomTag or Room base-class are allowed.")
            for item in content:
                if not isinstance(item, stack_type):
                    return False
                # Recursively validate fields of the named tuple in the stack
                for sub_field in stack_type._fields:
                    sub_content = getattr(item, sub_field)
                    if not cls.validate_field_contents(stack_type, sub_field, sub_content):
                        return False
                
            return True
        
        # Check if it's a SINGLETON_INT
        if SINGLETON_INT in field_annos:
            if not isinstance(content, (jnp.ndarray, jArray)):
                return False
            if content.dtype != jnp.int32:
                return False
            if content.shape != (1,):
                return False
            return True
            
        # Check if it's ROOM_SHAPED or CONSTANT_SHAPE
        if ROOM_SHAPED in field_annos or CONSTANT_SHAPE in field_annos:
            if not isinstance(content, (jnp.ndarray, jArray)):
                return False
            if content.dtype != jnp.int32:
                return False
            
            # These are usually arrays, further validation might depend on specific requirements
            return True

        return True
    
    
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
    
    @classmethod
    def _fetch_custom_annotations_per_field(cls, tup: type[NamedTuple]) -> dict[str, list[object]]:
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
    def _fields_for_nt(cls, tup: type[NamedTuple]) -> list[str]:
        hints = get_type_hints(tup, include_extras=True)
        fields: list[str] = list(hints.keys())
        return fields
    
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
        if stack_type not in cls._type_to_annos:
            return False

        annos: dict[str, list[object]] = cls._type_to_annos[stack_type]
        for field_annos in annos.values():
            if SINGLETON_INT not in field_annos:
                return False
            if NAMED_TUPLE_STACK in field_annos:
                return False
            if ROOM_SHAPED in field_annos or CONSTANT_SHAPE in field_annos:
                return False
        return True
        
    
    @classmethod
    def _validate_registered_named_tuples(self) -> None:
        """
        Named Tuple classes that are registered as the Content of a NamedTupleStack may only contain scalar fields (static and dynamic).
        Also, they may not be registred as either room or tag themselves.
        """
        for tup, field_annos in self._type_to_annos.items():
            for field, annos in field_annos.items():
                stack_type = self._get_anno_tuple_stack_type(annos)
                if stack_type is None:
                    continue
                if not self._check_eligible_named_tuple_stack_type(stack_type):
                    raise Exception(
                        f"NamedTuple {tup.__name__}, field {field}:: stack element type {stack_type.__name__} must be a registered non-room, non-tag NamedTuple containing only SINGLETON_INT fields."
                    )

    

    @classmethod
    def _serialise_named_tuple_in_stack(cls, tup_instance: NamedTuple) -> tuple[jArray, jArray]:
        tup_type = type(tup_instance)
        # Might as well do one last check for safety, although should have happened already...
        if not cls._check_eligible_named_tuple_stack_type(stack_type=type(tup_instance)):
            raise Exception(f"Received tuple of invalid stack-type {type(tup_instance).__name__}")
        static_fields: list[str] = []
        dynamic_fields: list[str] = []
        for field in sorted(list(cls._fields_for_nt(tup_type))):
            annos = cls.get_annotations(tup_type, field)
            if STATIC in annos:
                static_fields.append(field)
            elif DYNAMIC in annos:
                dynamic_fields.append(field)

        serialised_static: jArray = jnp.zeros((len(static_fields),), jnp.int32)
        serialised_dynamic: jArray = jnp.zeros((len(dynamic_fields),), jnp.int32)

        for idx, field in enumerate(static_fields):
            field_content = getattr(tup_instance, field)
            serialised_static = serialised_static.at[idx].set(field_content[0])
        for idx, field in enumerate(dynamic_fields):
            field_content = getattr(tup_instance, field)
            serialised_dynamic = serialised_dynamic.at[idx].set(field_content[0])
        return serialised_static, serialised_dynamic

    @classmethod
    def _serialise_field_content_for_storage(cls, tup: type[NamedTuple], field: str, content: jnp.ndarray|List[NamedTuple]) -> jArray:
        annos = cls.get_annotations(tup, field)
        if NAMED_TUPLE_STACK in annos:
            stack_type = cls._get_anno_tuple_stack_type(annos)
            if stack_type is None:
                raise Exception(f"Field {field} of {tup.__name__} is marked as NAMED_TUPLE_STACK but has no stack type annotation.")
            serialised_size = cls._serialised_named_tuple_size(stack_type)
            serialised_content = jnp.zeros((len(content), serialised_size), dtype=jnp.int32)
            for idx, item in enumerate(content):
                if not isinstance(item, stack_type):
                    raise Exception(
                        f"Field {field} of {tup.__name__} contains item of type {type(item).__name__}; expected {stack_type.__name__}."
                    )
                serialised_static, serialised_dynamic = cls._serialise_named_tuple_in_stack(item)
                serialised_item = jnp.concatenate([serialised_static, serialised_dynamic], axis=0)
                serialised_content = serialised_content.at[idx, ...].set(serialised_item)
            return serialised_content

        if not isinstance(content, (jnp.ndarray, jArray)):
            raise Exception(f"Field {field} of {tup.__name__} must be a jax array after room construction.")

        if ROOM_SHAPED in annos:
            # TODO: Pad ROOM_SHAPED fields to the canonical room size along the first two axes before stacking.
            pass

        return content
    
    def _serialize_nt_stack(self, contents: list[NamedTuple]) -> Tuple[jArray, jArray]:
        # TODO: This is shit
        if len(contents) == 0:
            return Exception("Received empty NamedTuple stack, this should not happen.")

        nt_type = type(contents[0])
        
        static_fields: list[str] = []
        dynamic_fields: list[str] = []
        for field in sorted(MontezumaRoomLayoutAuthority._fields_for_nt(nt_type)):
            annos = MontezumaRoomLayoutAuthority.get_annotations(nt_type, field)
            if STATIC in annos:
                static_fields.append(field)
            elif DYNAMIC in annos:
                dynamic_fields.append(field)
        static_fields = sorted(static_fields)
        dynamic_fields = sorted(dynamic_fields)
        static_stack: jnp.ndarray = None
        dynamic_stack: jnp.ndarray = None
        if len(static_fields) > 0:
            static_stack = jnp.zeros((len(contents), len(static_fields)), dtype=jnp.int32)
        if len(dynamic_fields) > 0:
            dynamic_fields = jnp.zeros((len(contents), len(dynamic_fields)), dtype=jnp.int32)

        for row_idx, stack_item in enumerate(contents):
            for col_idx, field in enumerate(static_fields):
                value = getattr(stack_item, field)
                static_stack = static_stack.at[row_idx, col_idx].set(value[0])

            for col_idx, field in enumerate(dynamic_fields):
                value = getattr(stack_item, field)
                dynamic_stack = dynamic_stack.at[row_idx, col_idx].set(value[0])

        return static_stack, dynamic_stack

    @classmethod
    def _stack_storage_field_contents(
        cls,
        tup: type[NamedTuple],
        field: str,
        room_contents: List[Tuple[int, jnp.ndarray|List[NamedTuple]]],
    ) -> Tuple[jArray|Tuple[jArray, jArray], int, Dict[int, int], Dict[int, int]]:
        """Generates storage data for a single field of a room-tag/

        Args:
            tup (type[NamedTuple]): Type of the named tuple the field belongs to.
            field (str): Name of the field.
            room_contents (List[Tuple[int, jnp.ndarray | List[NamedTuple]]]): List containing pairs of RoomID and the corresponding field-contents. 
                Works under the assumption that a room can only ever have one instance of any given tag. This seems reasonable...

        Returns:
            Tuple[jArray|Tuple[jArray, jArray], Dict[int, int], Dict[int, int]]: 
                                Stacked array containing contents of this field for all rooms. If given content is a NamedTupleStack,
                                    a tuple of arrays (static_stack, dynamic_stack) is returned. 
                                    If a NamedTuple type has either no static or no dynamic fields, one of the arrays is liable to be None, 
                                Dictionary mapping room id to offset in the stack (key for a room is only contained if the room actually has this tag...)
                                Dictionary mapping room id to padding applied along the first dimension. This is 0 if the field does not contain a NamedTupleStack

        """
        annos = cls.get_annotations(tup, field)
        
        
        room_offsets: Dict[int, int] = {}
        room_padding: Dict[int, int] = {}
        current_offset = 0

        if NAMED_TUPLE_STACK in annos:
            static_contents: list[jArray] = []
            dynamic_contents: list[jArray] = []
            max_slice_size = max((content.shape[0] for _, content in serialised_contents), default=0)
            padded_contents: List[jArray] = []
            stack_type = cls._get_anno_tuple_stack_type(annos)
            trailing_size = 0 if stack_type is None else cls._serialised_named_tuple_size(stack_type)

            for room_id, content in serialised_contents:
                padding = max_slice_size - content.shape[0]
                room_offsets[room_id] = current_offset
                room_padding[room_id] = padding
                current_offset += max_slice_size
                if padding > 0:
                    content = jnp.pad(content, ((0, padding), (0, 0)))
                padded_contents.append(content)

            if len(padded_contents) == 0:
                stacked_content = jnp.zeros((0, trailing_size), dtype=jnp.int32)
            else:
                stacked_content = jnp.concatenate(padded_contents, axis=0)
            return stacked_content, max_slice_size, room_offsets, room_padding

        stacked_parts: List[jArray] = []
        for room_id, content in serialised_contents:
            slice_size = content.shape[0]
            room_offsets[room_id] = current_offset
            room_padding[room_id] = 0
            current_offset += slice_size
            stacked_parts.append(content)

        if len(stacked_parts) == 0:
            stacked_content = jnp.zeros((0,), dtype=jnp.int32)
            return stacked_content, 0, room_offsets, room_padding

        stacked_content = jnp.concatenate(stacked_parts, axis=0)
        return stacked_content, room_offsets, room_padding

    @classmethod
    def build_layout(cls, layout: NewPyramidLayout) -> RoomPersistenceStorage:
        if cls._room_named_tuple is None:
            raise Exception("Cannot build layout storage without a registered base room NamedTuple.")

        storage = RoomPersistenceStorage()
        storage.base_room_static_fields = {}
        storage.base_room_dynamic_fields = {}
        storage.tag_static_fields = {}
        storage.tag_dynamic_fields = {}
        storage.base_room_field_slice_size = {}
        storage.tag_field_slice_size = {}
        storage.room_field_id_offset = {}
        storage.tag_field_room_id_offset = {}
        storage.base_room_ntstack_padding = {}
        storage.tag_field_ntstack_padding = {}

        rooms: List[NewRoom] = [layout.rooms[room_id] for room_id in sorted(layout.rooms.keys())]
        for room in rooms:
            room._build_fields()

        base_fields = cls._fields_for_nt(cls._room_named_tuple)
        for field in base_fields:
            room_contents = [(room.room_id, room.base_class_fields[field]) for room in rooms]
            stacked_content, slice_size, room_offsets, room_padding = cls._stack_storage_field_contents(
                cls._room_named_tuple,
                field,
                room_contents,
            )
            annos = cls.get_annotations(cls._room_named_tuple, field)
            if STATIC in annos:
                storage.base_room_static_fields[field] = stacked_content
            else:
                storage.base_room_dynamic_fields[field] = stacked_content
            storage.base_room_field_slice_size[field] = slice_size
            storage.room_field_id_offset[field] = room_offsets
            storage.base_room_ntstack_padding[field] = room_padding

        present_tags = sorted({tag for room in rooms for tag in room.tags}, key=lambda tag: tag.__name__)
        for tag in present_tags:
            tag_name = tag.__name__
            storage.tag_static_fields[tag_name] = {}
            storage.tag_dynamic_fields[tag_name] = {}
            storage.tag_field_slice_size[tag_name] = {}
            storage.tag_field_room_id_offset[tag_name] = {}
            storage.tag_field_ntstack_padding[tag_name] = {}

            for field in cls._fields_for_nt(tag):
                room_contents = [
                    (room.room_id, room.tag_fields[tag][field])
                    for room in rooms
                    if tag in room.tags
                ]
                stacked_content, slice_size, room_offsets, room_padding = cls._stack_storage_field_contents(
                    tag,
                    field,
                    room_contents,
                )
                annos = cls.get_annotations(tag, field)
                if STATIC in annos:
                    storage.tag_static_fields[tag_name][field] = stacked_content
                else:
                    storage.tag_dynamic_fields[tag_name][field] = stacked_content
                storage.tag_field_slice_size[tag_name][field] = slice_size
                storage.tag_field_room_id_offset[tag_name][field] = room_offsets
                storage.tag_field_ntstack_padding[tag_name][field] = room_padding

        return storage
        