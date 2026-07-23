from copy import deepcopy
#[codeCombination]
def _find_combinations(items, group_size):
	if group_size >= len(items):
		return [deepcopy(items)]

	results = []
	_find_combinations_helper(items, group_size, 0, [], results)

	return results

def _find_combinations_helper(items, group_size, cur_idx, cur_group, results):
	if len(cur_group) == group_size:
		results.append(deepcopy(cur_group))
		return

	while cur_idx < len(items):
		cur_group.append(items[cur_idx])
		_find_combinations_helper(items, group_size, cur_idx + 1, cur_group, results)
		cur_group.pop()
		cur_idx += 1



# 1 DataStructure with O(1) Operations [GetRandom]
"""
In order to do remove O(1),
Imagine we are store value in a list but
Use 2 map to record value to index, and index to value

When insert, insert to end of list and update maps
When remove, if the value is not the last, swap position with last index value and
remove from maps
When random, use random to get a random index
"""

import random

class RandomizedSet:

    def __init__(self):
        self.index_to_value = {}
        self.value_to_index = {}

    def insert(self, val: int) -> bool:
        if val in self.value_to_index:
            return False

        index = len(self.index_to_value)

        self.index_to_value[index] = val
        self.value_to_index[val] = index
    
        return True
        

    def remove(self, val: int) -> bool:
        if val not in self.value_to_index:
            return False
        
        index = self.value_to_index[val]

        # if at the end, just remove
        if index == len(self.index_to_value) - 1:
            del self.index_to_value[index]
            del self.value_to_index[val]
        else:
            # else swap with last one
            last_index = len(self.index_to_value) - 1
            last_val = self.index_to_value[last_index]

            self.index_to_value[index] = last_val
            self.value_to_index[last_val] = index

            del self.index_to_value[last_index]
            del self.value_to_index[val]
        
        return True

    def getRandom(self) -> int:
        index = random.randint(0, len(self.index_to_value) - 1)
        return self.index_to_value[index]
        

# GetRandom With Duplication
"""
In order to do remove O(1),
Imagine we are store value in a list but
Use 2 map to record value to indices, and index to value

When insert, insert to end of list and update maps
When remove, if the value is not the last, swap position with last index value and
remove from maps
When random, use random to get a random index
"""

import random
from collections import defaultdict

class RandomizedCollection:
    def __init__(self):
        self.index_to_value = {}
        self.value_to_indices = defaultdict(set)
        
    def insert(self, val: int) -> bool:
        result = val not in self.value_to_indices

        index = len(self.index_to_value)
        self.value_to_indices[val].add(index)
        self.index_to_value[index] = val

        return result

    def remove(self, val: int) -> bool:
        if val not in self.value_to_indices:
            return False
        
        # pop a index
        index = self.value_to_indices[val].pop()
        # if no more index, del the val from map
        if not self.value_to_indices[val]:
            del self.value_to_indices[val]

        # if last one, remove
        if index == len(self.index_to_value) - 1:
            del self.index_to_value[index]
        else:
            # else swap with last one
            last_index = len(self.index_to_value) - 1
            last_value = self.index_to_value[last_index]

            self.value_to_indices[last_value].add(index)
            self.index_to_value[index] = last_value

            del self.index_to_value[last_index]
            self.value_to_indices[last_value].discard(last_index)
        
        return True

    def getRandom(self) -> int:
        index = random.randint(0, len(self.index_to_value) - 1)

        return self.index_to_value[index]
        


# 2 Concurrent users
# - meeting Rooms
# - merge interval - lc 56

from queue import PriorityQueue
from bisect import bisect_left

class IntervalCounter:
    def __init__(self, intervals: List[List[int]]):
        # (point, overlap)
        self.overlap_points = self._process_overlap(intervals)

    def query(self, x: int) -> int:
        # bisect_left return insertion index.
        index = bisect_left([p[0] for p in self.overlap_points], x)
        if index == len(self.overlap_points) or self.overlap_points[index][0] > x:
            index -= 1
        result = self.overlap_points[index][1] if index >= 0 else 0

        print(f'query-{x}')
        print(f'self.overlap_points-{self.overlap_points}')
        print(f'index-{index}')

        return result

    def _binary_search(self, target):
        """
        Find the rightmost point where overlap_point[index] <= target
        """
        left, right = 0, len(self.overlap_points) - 1
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            if self.overlap_points[mid][0] <= target:
                left = mid + 1
                ans = mid
            else:
                right = mid - 1

        return ans

    def _process_overlap(self, intervals):
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        # record end time
        pq = PriorityQueue()

        # (point, overlap)
        overlap_points = []

        for interval in sorted_intervals:
            start, end = interval[0], interval[1]
            # kick passed intervals
            # when add ends, since end is inclusive, we add end+1 to overlap list
            while not pq.empty() and pq.queue[0] < start:
                pass_end = pq.get()
                self._update_overlap_points(pass_end + 1, overlap_points, pq.qsize())
            
            pq.put(end)
            # check if multiple start at same point
            self._update_overlap_points(start, overlap_points, pq.qsize())

        # process the ends, check if mutliple ends at same point
        # when add ends, since end is inclusive, we add end+1 to overlap list
        while not pq.empty():
            point = pq.get()
            self._update_overlap_points(point + 1, overlap_points, pq.qsize())

        return overlap_points

    def _update_overlap_points(self, point, overlap_points, size):
        if overlap_points and point == overlap_points[-1][0]:
                overlap_points.pop()    
        overlap_points.append((point, size))



# meeting room I - can user attend all meetings
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

        for idx in range(1, len(sorted_intervals)):
            if sorted_intervals[idx - 1][1] > sorted_intervals[idx][0]:
                return False
        
        return True

# meeting room II - min meeting rooms
# [meetRoomCount]
"""
Use a priority queue (min-heap) to record current number of meeting rooms
items in pq is the current end time of meetings

sort all meetings
when encounter a new meeting, if its start time is smaller than 
smallest end time, we need a new room. add its end time to pq
otherwise, keep poping until end time is smaller than cur start time and add
its end time to pq

Time complexity - O(nlgn)
"""
from queue import PriorityQueue


class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        room_count = 0
        pq = PriorityQueue()

        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

        for interval in sorted_intervals:
            start, end = interval[0], interval[1]

            # compare end times and pop
            while not pq.empty() and pq.queue[0] <= start:
                pq.get()

            pq.put(end)
            room_count = max(pq.qsize(), room_count)
        
        return room_count

# meeting room III - find room take most meetings
# [mostBook]
"""
Use a counter record each room meeting count
Use a PriorityQueue (min-heap) to record current busy room
The item is (end_time, room_id)
Use a set to record free_room

When encounter a meeting,
Try to pop out rooms whose end_time <= meeting_start and put them into 
free_room
If no free_room, shift cur meeting time to smallest end_time room in pq
update that room end_time to end_time + meet_end_time - meet_start_time
if free_room, find a free_room id

counter_map[room_id] += 1

and find the max

Time Complexity - O(mlgm + mlgn + nlgn) meetings m | rooms n
- sort meetings mlgm
- pq with n rooms at most mlgn
- find max count on rooms nlgn
"""
from queue import PriorityQueue
from collections import defaultdict

class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        free_room = PriorityQueue()
        pq = PriorityQueue()
        counter = defaultdict(int)

        for i in range(n):
            free_room.put(i)

        sorted_meetings = sorted(meetings, key=lambda x: (x[0], x[1]))

        for meeting in sorted_meetings:
            start, end = meeting[0], meeting[1]

            while not pq.empty() and pq.queue[0][0] <= start:
                free_room.put(pq.get()[1])
            
            if not free_room.empty():
                room_id = free_room.get()
                pq.put((end, room_id))
            else:
                last_end, room_id = pq.get()
                pq.put((last_end + end - start, room_id))
            
            counter[room_id] += 1

        # if count is same, output room id with smallest one
        return -sorted([(count, -room_id) for room_id, count in counter.items()])[-1][1]

# merge interval
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        result = []

        cur_start, cur_end = sorted_intervals[0][0], sorted_intervals[0][1]

        for idx in range(1, len(sorted_intervals)):
            start, end = sorted_intervals[idx][0], sorted_intervals[idx][1]
            if start > cur_end:
                result.append([cur_start, cur_end])
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        
        result.append([cur_start, cur_end])

        return result

# [IntvervalCounter]
# [OverlapCounter]

from queue import PriorityQueue

class IntervalCounter:
    def __init__(self, intervals: List[List[int]]):
        # (point, overlap)
        self.overlap_points = self._process_overlap(intervals)

    def query(self, x: int) -> int:
        index = self._binary_search(x)
        result = self.overlap_points[index][1] if index >= 0 else 0

        return result

    def _binary_search(self, target):
        """
        Find the rightmost point where overlap_point[index] <= target
        """
        left, right = 0, len(self.overlap_points) - 1
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            if self.overlap_points[mid][0] <= target:
                left = mid + 1
                ans = mid
            else:
                right = mid - 1

        return ans

    def _process_overlap(self, intervals):
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        # record end time
        pq = PriorityQueue()

        # (point, overlap)
        overlap_points = []

        for interval in sorted_intervals:
            start, end = interval[0], interval[1]
            # kick passed intervals
            # when add ends, since end is inclusive, we add end+1 to overlap list
            while not pq.empty() and pq.queue[0] < start:
                pass_end = pq.get()
                self._update_overlap_points(pass_end + 1, overlap_points, pq.qsize())
            
            pq.put(end)
            # check if multiple start at same point
            self._update_overlap_points(start, overlap_points, pq.qsize())

        # process the ends, check if mutliple ends at same point
        # when add ends, since end is inclusive, we add end+1 to overlap list
        while not pq.empty():
            point = pq.get()
            self._update_overlap_points(point + 1, overlap_points, pq.qsize())

        return overlap_points

    def _update_overlap_points(self, point, overlap_points, size):
        if overlap_points and point == overlap_points[-1][0]:
                overlap_points.pop()    
        overlap_points.append((point, size))


# 3 Content management System
# Versioned File 
"""
For Simple version, it is hard to identify a version_id belongs to which key
Also anyone can put a random version id to try to delete or read a version of a key
"""
from collections import defaultdict

class ContentManagementSimple:
	def __init__(self):
		# path to map of versions to content as list 
		self.history_version_content = defaultdict(list)

	# if create twice should fail
	def create(self, path, content):
		if path in self.history_version_content:
			raise ValueError(f'path-{path} already exists')

		self.history_version_content[path].append(content)

	# if only update without create should fail
	def update(self, path, content):
		if path not in self.history_version_content:
			raise ValueError(f'path-{path} not exists')

		self.history_version_content[path].append(content)

	def delete(self, path):
		if path not in self.history_version_content:
			return False

		del self.history_version_content[path]

		return True

	def read(self, path):
		if path not in self.history_version_content:
			return None
		return self.history_version_content[path][-1]

	def read_version(self, path, version):
		if path not in self.history_version_content or version >= len(self.history_version_content[path]):
			return None

		return self.history_version_content[path][version]



from collections import defaultdict, OrderedDict

class EntityMeta:
	def __init__(self, path):
		self.path = path
		self.version_idx = 0
		self.cur_version = None
		# version idx maps to content
		self.versions = OrderedDict()

	def add_content(self, content):
		idx = self.version_idx
		self.version_idx += 1

		self.cur_version = idx
		self.versions[idx] = content

		return self.cur_version

	def delete_version(self, version_idx):
		if version_idx not in self.versions:
			raise ValueError(f'version_idx - {version_idx} not exists')

		del self.versions[version_idx]

		# if it is current version, we need to move to next ver
		if self.cur_version == version_idx:
			most_recent_version = next(reversed(self.versions.keys()))
			self.cur_version = most_recent_version


class ContentManagementProduction:
	def __init__(self):
		# path -> EntityMeta
		self.metadata = {}

	def add(self, path, content):
		if path not in self.metadata:
			self.metadata[path] = EntityMeta(path)

		ver = self.metadata[path].add_content(content)

		return ver

	def delete(self, path):
		if path not in self.metadata:
			return False

		del self.metadata[path]

		return True

	def delete_by_version(self, path, version_id):
		if path not in self.metadata or version_id not in self.metadata[path].versions:
			return False

		self.metadata[path].delete_version(version_id)

		return True


	def read(self, path):
		if path not in self.metadata:
			return None

		return self.metadata[path].versions[self.metadata[path].cur_version]

	def read_version(self, path, version_id):
		if path not in self.metadata or version_id not in self.metadata[path].versions:
			return None

		return self.metadata[path].versions[version_id]


cmS = ContentManagementSimple()
cmS.add('123/321/abc', 'abc')
print('------------------------')
print(cmS.read('123/321/abc'))
print(cmS.read('123/321/abc') == 'abc')

print('------------------------')
cmS.add('123/321/abc', 'abcd')
print(cmS.read('123/321/abc'))
print(cmS.read('123/321/abc') == 'abcd')

print('------------------------')
print(cmS.read_version('123/321/abc', 0))
print(cmS.read_version('123/321/abc', 0) == 'abc')

print('------------------------')
cmS.delete('123/321/abc')
print(cmS.read('123/321/abc'))
print(cmS.read('123/321/abc') is None)


# followup
# 1 - what steps do you expect to productization?
"When deploy the system, our CI/CD pipeline should trigger an integration test"
"Integeration test for add multiple version and delete current version"
"Observability - latency of each api, logging"
# 2 - how would you architect the system
"Store meta data in SQL and content in S3"


# 4 Timed Caching
# [ExpireCache]

"""
1. when talking about cache, mem leak should always be a concern
2. I assume we have limited mem size, and we need to prevent OOM
3. whenever the api is called, we should clear those expired keys
4. If there are keys never expire, but we keep inserting new keys, i think we 
need to limit the cache size and evicted the less used ones. like LRU style

to know which key expire first. I can use a priority queue. But 
I also need to remove items when update LRU. I could use SortedDict(treemap) to keep the order 
and able to do remove
to mimic LRU behavior. I could use SortedDict which keep the order of inserted keys
"""
import time
from sortedcontainers import SortedDict
from collections import OrderedDict

class ExpireCache:
	def __init__(self, capacity):
		self.capacity = capacity
		# ttl -> set(keys), sorted by ttl ascending
		self.ttl_to_keys = SortedDict()
		# key -> (value, ttl), ordered by inserted order
		self.key_to_values = OrderedDict()

	def set(self, key, value, ttl):
		# time.time() is seconds as float
		cur_time = int(time.time() * 1000)
		self._clean_expired_keys(cur_time)

		print(cur_time)

		# if in the cache already, refresh it
		if key in self.key_to_values:
			_, original_ttl = self.key_to_values[key]
			del self.key_to_values[key]

			if original_ttl in self.ttl_to_keys:
				self.ttl_to_keys[original_ttl].discard(key)

		# update all maps
		new_ttl_ts = cur_time + ttl
		self.key_to_values[key] = (value, new_ttl_ts)
		
		if new_ttl_ts not in self.ttl_to_keys:
			self.ttl_to_keys[new_ttl_ts] = set()
		self.ttl_to_keys[new_ttl_ts].add(key)

		# clean LRU
		self._clean_lru()

	def get(self, key):
		cur_time = int(time.time() * 1000)
		self._clean_expired_keys(cur_time)

		if key not in self.key_to_values:
			return None

		value, _ = self.key_to_values[key]

		# resset position of key to last in lru
		self.key_to_values.move_to_end(key)

		return value

	def _clean_lru(self):
		while len(self.key_to_values) > self.capacity:
			# pop the first inserted key
			key, value_tuple = self.key_to_values.popitem(last=False)
			ttl_ts = value_tuple[1]

			self.ttl_to_keys[ttl_ts].discard(key)
			if not self.ttl_to_keys[ttl_ts]:
				del self.ttl_to_keys[ttl_ts]

	def _clean_expired_keys(self, cur_time):
		for ttl in self.ttl_to_keys.keys():
			if ttl < cur_time:
				expired_keys = self.ttl_to_keys[ttl]
				del self.ttl_to_keys[ttl]

				for key in expired_keys:
					del self.key_to_values[key]
			else:
				# keys are sorted
				break


from unittest import mock

# - set -> get / get expired
# - continue set with size beyong capacity and get evicted key
# - set existing key and evicted other oldest key
# - get existing key and evicted other oldest key
# - continue set with key expire thus insert new key has room and none evicted

times = [1.000, 1.050, 2.500]
with mock.patch('time.time', side_effect=times):
	cache = ExpireCache(5)
	cache.set('a', '1', 100)

	assert cache.get('a') == '1'
	assert cache.get('a') is None

# continue set with size beyong capacity and get evicted key
times = [1.000, 1.000, 1.000, 1.000, 1.050, 1.050, 1.150, 1.150]
with mock.patch('time.time', side_effect=times):
	cache = ExpireCache(3)
	cache.set('a', '1', 100)
	cache.set('b', '2', 100)
	cache.set('c', '3', 100)
	cache.set('d', '4', 100)

	assert cache.get('a') is None
	assert cache.get('b') == '2'
	assert cache.get('c') is None # -> all expired
	assert cache.get('b') is None # -> all expired


# set existing key and evicted other oldest key
times = [1.000, 1.000, 1.000, 1.050, 1.050, 1.050, 1.050]
with mock.patch('time.time', side_effect=times):
	cache = ExpireCache(3)
	cache.set('a', '1', 100)
	cache.set('b', '2', 100)
	cache.set('c', '3', 100)
	cache.set('a', '1', 100)
	cache.set('d', '4', 100) # -> b should be evicted

	assert cache.get('b') is None
	assert cache.get('a') == '1'

# get existing key and evicted other oldest key
times = [1.000, 1.000, 1.000, 1.050, 1.050, 1.050, 1.050]
with mock.patch('time.time', side_effect=times):
	cache = ExpireCache(3)
	cache.set('a', '1', 100)
	cache.set('b', '2', 100)
	cache.set('c', '3', 100)
	
	assert cache.get('a') == '1'
	
	cache.set('d', '4', 100) # -> b should be evicted

	assert cache.get('b') is None

# continue set with key expire thus insert new key has room and none evicted
times = [1.000, 1.000, 1.000, 1.050, 1.150, 1.150]
with mock.patch('time.time', side_effect=times):
	cache = ExpireCache(3)
	cache.set('a', '1', 100)
	cache.set('b', '2', 200)
	cache.set('c', '3', 300)
	
	assert cache.get('a') == '1'
	
	cache.set('d', '4', 100) # -> a expired

	assert cache.get('b') == '2'


# Event Logger with Rate Limiting
"""
If logs sometimes arrives not in time order, thats fine.
We have apporximate behavior
"""
class RateLimiter:
	def __init__(self, capacity, ttl):
		self.ttl = ttl
		self.capacity = capacity
		# ttl -> set(keys), sorted by ttl ascending
		self.ttl_to_keys = SortedDict()
		# keys -> ttl
		self.keys_to_ttl = OrderedDict()

	def print_event(self, event_name, timestamp):
		self._clean_expired_keys(timestamp)

		result = ""

		# if in the cache already and not expired, refresh it
		if key in self.keys_to_ttl:
			# not print event
			original_ttl = self.keys_to_ttl[key]
			del self.key_to_values[key]

			if original_ttl in self.ttl_to_keys:
				self.ttl_to_keys[original_ttl].discard(key)
		else:
			result = event_name

		# update all maps
		new_ttl_ts = timestamp + self.ttl
		self.keys_to_ttl[key] = (value, new_ttl_ts)
		
		if new_ttl_ts not in self.ttl_to_keys:
			self.ttl_to_keys[new_ttl_ts] = set()
		self.ttl_to_keys[new_ttl_ts].add(key)

		# clean LRU
		self._clean_lru()

		return result

	def _clean_lru(self):
		while len(self.key_to_values) > self.capacity:
			# pop the first inserted key
			key, ttl_ts = self.key_to_values.popitem(last=False)

			self.ttl_to_keys[ttl_ts].discard(key)
			if not self.ttl_to_keys[ttl_ts]:
				del self.ttl_to_keys[ttl_ts]

	def _clean_expired_keys(self, cur_time):
		for ttl in self.ttl_to_keys.keys():
			if ttl < cur_time:
				expired_keys = self.ttl_to_keys[ttl]
				del self.ttl_to_keys[ttl]

				for key in expired_keys:
					del self.key_to_values[key]

# 5 Recommendation
# netlifx home page has a list of shelf，每个 shelf 有一堆 title。要在 viewport（每个 shelf 最多显示 X 个）里做dedupe。
# 前 X 位跨 shelf 去重，之后只在各自 shelf 内去重
def dedupe(titles: list[list[int]], x: int):
  global_visited = set()
  ret = []
  for shelf in titles:
    current = []
    local_visited = set()
    for title in shelf:
      if len(current) < x and title not in global_visited:
        current.append(title)
        global_visited.add(title)
        local_visited.add(title)
      elif len(current) >= x and title not in local_visited:
        current.append(title)
        local_visited.add(title)
    ret.append(current)
  return ret


# friends with common movies
# [commonMovieSuffix]
# q1 - find friends with common suffix k movies
from collections import defaultdict

class Solution:
    def findFriends(self, customerIds: List[str], movies: List[List[str]], k: int) -> List[List[str]]:
        """
        Keep a map where key - movie suffix, value - list of customer id
        movie suffix = "-".join(sorted(movers[-k:]))
        Time complexity - O(m*k)
        """

        movie_group = defaultdict(list)

        for c_id, movie_list in enumerate(movies):
            movie_suffix = '-'.join(sorted(movie_list[-k:]))
            movie_group[movie_suffix].append(customerIds[c_id])
        
        result = []
        for group in movie_group.values():
            if len(group) >= 2:
                result.append(group)

        return result

# q2 - find pairs in suffix k movies at least m movie matched

"""
Build a revert index of movie -> list of user id
movie are suffix movies of each users
calculate pair count by shared movies
Time complexity: O(m*k + n*s^2) - n common movies of suffix, s - average length of user shared a movie
"""
from collections import defaultdict

class Solution:
    def findFriendsWithM(self, customerIds: List[str], movies: List[List[str]], k: int, m: int) -> List[List[str]]:
        # build movie -> [user] revert index
        movie_to_user = defaultdict(list)
        for c_id, movie_list in enumerate(movies):
            movie_suffix = movie_list[-k:]
            for movie in movie_suffix:
                movie_to_user[movie].append(customerIds[c_id])

        # count user pair 
        user_pair_count = defaultdict(int)
        for users_list in movie_to_user.values():
            for u_1 in range(0, len(users_list) - 1):
                for u_2 in range(u_1 + 1, len(users_list)):
                	# maybe sort the u1 and u2 here to keep the order
                    user_pair_count[(users_list[u_1], users_list[u_2])] += 1

        result = []
        for u_pair, count in user_pair_count.items():
            if count >= m:
                result.append(list(u_pair))

        return result

# recommend movies
from queue import PriorityQueue

class Solution:
    def bestGroupIndex(self, similarityScores: List[float], groups: List[List[int]], k: int) -> int:
        """
        Use a min-heap for each group to calculate the top scores and total scores
        so finding top k movie is O(mlgk) - each gourp has m movies
        """
        best_score_idx = -1
        best_score = -1

        for g_idx, group in enumerate(groups):
            pq = PriorityQueue()

            for movie in group:
                score = similarityScores[movie]
                if pq.qsize() >= k and pq.queue[0] < score:
                    pq.get()
                    pq.put(score)
                elif pq.qsize() < k:
                    pq.put(score)

            total_score = 0
            while not pq.empty():
                total_score += pq.get()

            if total_score > best_score:
                best_score = total_score
                best_score_idx = g_idx
        
        return best_score_idx



# 6 Events deduplication
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class Event:
	name: str
	ts_in_seconds: int


class Solution:
	def __init__(self):
		self.events_to_ts = OrderedDict()

	def process_event(self, event):
		result = None
		if event.name not in self.events_to_ts or event.ts_in_seconds - self.events_to_ts[event.name] >= 10:
			result = event.name

		self.events_to_ts[event.name] = event.ts_in_seconds
		self.events_to_ts.move_to_end(event.name)

		# clean up
		for key in self.events_to_ts.keys():
			if self.events_to_ts[key] < event.ts_in_seconds - 10:
				del self.events_to_ts[key]
			else:
				break

		return result


events = [
	Event('foo', 1000000),
	Event('bar', 1000002),
	Event('baz', 1000005),
	Event('foo', 1000008),
	Event('bar', 1000013),
	Event('baz', 1000016),
	Event('foo', 1000019),
]

sol = Solution()
result = []

for event in events:
	output = sol.process_event(event)
	if output:
		result.append(output)

assert result == ['foo', 'bar', 'baz', 'bar', 'baz', 'foo']





# 7 job management
# 

# 8 parallel courses
"""
Use topological sort to traverse the graph when a course indegree is 0
Basically a BFS, each wave is a semester

Time complexity: O(N + E) n - count of course | e - edges between courses
we visited each edge once
"""
# parallel courses I - minimum semester
# [minSemester][parallelCourses]
from collections import defaultdict
from queue import deque

class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        semester = 0
        graph = defaultdict(set)
        indegree = defaultdict(int)

        for rel in relations:
            prev = rel[0]
            next_c = rel[1]
            
            graph[prev].add(next_c)
            indegree[next_c] += 1

        visited = 0
        queue = deque()
        for c in range(1, n + 1):
            if c not in indegree or not indegree[c]:
                queue.append(c)
        
        while queue:
            semester += 1
            cur_size = len(queue)

            for _ in range(cur_size):
                cur_c = queue.popleft()
                visited += 1

                if cur_c not in graph:
                    continue
                for next_c in graph[cur_c]:
                    indegree[next_c] -= 1
                    # if 0 indegree
                    if not indegree[next_c]:
                        queue.append(next_c)
        
        if visited == n:
            return semester
        return -1


# parallel course II - only take k course per semester
# [KCourse ParallelCourseK]

from itertools import combinations

from collections import defaultdict
from queue import deque

def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
	"""
	There are totally 2^n states for n courses
	Each state can have multiple next states since next available course can have different combinations
	I think we can use BFS on states to find the shortest step to visit all node in the map
	Each entity in the BFS queue is a state on how many courses we take
	We can use a bitmap to record currently status of node visited or not
	1 is visited, 0 is not
	And a map record indegree and dependencies. 
	the key is node, value is a bitmap where 1s are prerequired
	so to check if a node's all requirements are visited, we can do 
	pre[i]&cur_state == pre[i] then we can visit this node

	We start with 0 visited and 0 step
	Also need a map to record the if the state we already visited or not

	Time complexity: O(2^n*C(m,k))
	"""
	pre = defaultdict(int)
	visited = set()

	for rel in relations:
	    pre_c = rel[0] - 1
	    next_c = rel[1] - 1
	    # build prerequirements bitmap
	    pre[next_c] += 1 << pre_c

	end_state = (1<<n) - 1

	queue = deque()
	# start with 0 node visited, 0 steps
	queue.append((0, 0))
	while queue:
	    cur_state, step = queue.popleft()
	    # if all node visited
	    if cur_state == end_state:
	        return step

	    next_visit = []
	    for i in range(n):
	        # indegree is not 0
	        if pre[i] & cur_state != pre[i]:
	            continue
	        # current course is visited
	        if cur_state & (1 << i) != 0:
	            continue
	        next_visit.append(i)

	    for comb in itertools.combinations(next_visit, min(len(next_visit), k)):
	        next_state = cur_state
	        for course in comb:
	            next_state += 1 << course
	        if next_state not in visited:
	            visited.add(next_state)
	            queue.append((next_state, step + 1))

	# in the end in we drain all options but still no hit the target we are not able to finish
	return -1





"""
Imagine we are traverse through the graph.
Each time, we need to figure out next wave of candidates.
If count of candidate > k, we will need to find all combinations and iterate through
If count of candidate <= k, we can take them all
Then we need to update indegree map and calculate next wave of candidates

Since we need to iterat through candidate groups. DFS on each candidate group can be a 
choice. And each time when encounter the end (no more candidate and all courses taken), return steps
and on each level return the min(all steps)

For DFS, we can also remember the path we already calculated. To record the status of course taken,
we can use bitmap 1 is not taken 0 is taken. 
to quickly know if any course taken, ie, for course i, mask & (1 << i) if 1 then non-taken, if 0 then taken 

In Python, we can use itertools.combinations as helper to find combinations
Also the course are 1-index, to use mask we need to make them 0-indexed

Time complexity: O(2^n * (n + combination(M, K))) - we have 2^n masks, that is the sub-path for DFS
on each path, we scan n nodes for next group, and if we have m nodes available, we find combinations with k group size
combination(M, K) = M! / (K! * (M-K)!)
"""

from itertools import combinations
from copy import deepcopy

from collections import defaultdict

class Solution:
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
        graph = defaultdict(set)
        indegree = defaultdict(int)

        for rel in relations:
            # convert to 0 index
            prev = rel[0] - 1
            next_c = rel[1] - 1

            graph[prev].add(next_c)
            indegree[next_c] += 1

        # 0-indexed mask
        # if 3 course, we want 111, (1 << 3) - 1 = 7 -> 111
        # or mask = ['1'] * n
        mask = (1 << n) - 1
        step_records = {}

        return self.dfs(mask, indegree, n, k, graph, step_records)


    def dfs(self, mask, indegree, n, k, graph, step_records):
        # check if mask == 0 if true we taken all courses return 0 step
        if not mask:
            return 0

        if mask in step_records:
            return step_records[mask]

        # find candidates from mask whose indegree is 0
        nodes = [i for i in range(n) if mask & 1 << i and indegree[i] == 0]

        result = float('inf')
        # loop through node groups
        for k_nodes in combinations(nodes, min(k, len(nodes))):
            cur_mask = mask
            cur_indegree = deepcopy(indegree)

            for node in k_nodes:
                # mark the node taken, the XOR make same postition bit turn to 0
                # since both are 1s.
                cur_mask ^= 1 << node
                
                for neighbor in graph[node]:
                    cur_indegree[neighbor] -= 1
            step_records[cur_mask] = self.dfs(cur_mask, cur_indegree, n, k, graph, step_records)

            result = min(result, 1 + step_records[cur_mask])
        
        return result
        

# parallel course III - course take time
# [parallelCourseTime]
"""
Topological sort
And use a max_time record max time takes to reach each node
as a node pre nodes can be taken same time but cost max(pre_node_time)
and max_time[node] = max(time[node] + max_time[pre_node])

Time complexity: O(E + N) e - edges, n - total courses
"""

from collections import defaultdict
from queue import deque

class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        graph = defaultdict(set)
        indegree = defaultdict(int)

        # course are 1 indexed,
        # convert to 0 indexed
        for rel in relations:
            prev = rel[0] - 1
            next_c = rel[1] - 1

            graph[prev].add(next_c)
            indegree[next_c] += 1
        
        queue = deque()
        max_time = [0] * n
        for c in range(n):
            if c not in indegree or not indegree[c]:
                queue.append(c)
                max_time[c] = time[c]
        
        while queue:
            cur_c = queue.popleft()
            if cur_c not in graph:
                continue

            for neighbor in graph[cur_c]:
                max_time[neighbor] = max(max_time[neighbor], max_time[cur_c] + time[neighbor])
                indegree[neighbor] -= 1

                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        return max(max_time)
        



# 9 budget cap
# [BudgetCap]

"""
To calculate total budget after capped,
total = cap * [capped show counter] + [remain uncapped show total]

Thus we can sort the show budgets.
Maintain a prefix sum = sum(budget[:cur])
total = prefix_sum + cap * (len - cur)

so we can find cap, as long as cap > 0 and <= budget[cur], we find it
"""

def find_budget_cap(costs, total_budget):
	sorted_cost = sorted(costs)

	prefix_sum = 0

	for idx in range(len(sorted_cost)):
		cur_cap = (total_budget - prefix_sum) / (len(sorted_cost) - idx)
		if 0 < cur_cap <= sorted_cost[idx]:
			return cur_cap

		prefix_sum += sorted_cost[idx]

	# If total_budget is >= sum(requests), no cap is needed.
	return sorted_cost[-1]


# 10 string to integer
"""
When encounter a digit, convert it to ascii and minus ascii '0'
in python use ord('x') for ascii code
if ord(a) - ord('0') > 9, then it is not a digit
"""
class Solution:
    def myAtoi(self, s: str) -> int:
        striped = s.strip()

        result = 0
        sign = True
        index = 0

        MAX_INT = 2**31 - 1
        MIN_INT = -2**31

        if not striped:
            return 0

        if striped[0] in set(['-', '+']):
            sign = striped[0] == '+'
            index = 1

        while index < len(striped):
            c = striped[index]
            digit = ord(c) - ord('0')
            if digit > 9 or digit < 0:
                break

            if (MAX_INT - digit) // 10 < result:
                return MAX_INT if sign else MIN_INT

            result = result * 10 + digit
            index += 1
        
        return result if sign else -result

# ------------------------------------

# 11 contain duplicates
# [containDuplicate]
# contain duplicates I
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums_set = set()

        for num in nums:
            if num in nums_set:
                return True
            else:
                nums_set.add(num)
        
        return False

# contain duplicates II - window size K
"""
Keep a window of size K
when size > k,
move low forward and keep poping
if nums[high] in the window, return True
otherwise move high and put nums[high] in window
"""
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        window = set()
        low = 0
        high = 0

        while high < len(nums):
            if len(window) > k:
                window.discard(nums[low])
                low += 1
            elif nums[high] in window:
                return True
            else:
                window.add(nums[high])
                high += 1
        
        return False

# contain duplicates III - bucket sort
"""
We need to quickly find elements within window k that might have diff < value_diff
to do this, we can put elements into bucket where each bucket size as value_diff
so to find if there are others have value_diff we check its own and neighbor buckets
we can say each bucket only has 1 element. since if there are 2 element in 1 bucket,
meaning we find the pair

bucket size [0, t], [t + 1, 2t + 1] ...
size is t + 1
bucket index = value // size (if value = t, bucket index = 0)

loop through the elements
1. get the bucket id (ele // bucket_size)
2. if bucket_id in current buckets, we find the pair
3. check neighbor buckets
4. if none found, put into bucket
5. maintain window size, if index >= k, remove nums[i - k]
"""
class Solution:
    def _get_bucket_id(self, value, size):
        return value // size

    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        buckets = {}
        size = valueDiff + 1

        for index in range(len(nums)):
            bucket_id = self._get_bucket_id(nums[index], size)
            if bucket_id in buckets:
                return True
            if bucket_id - 1 in buckets and abs(buckets[bucket_id - 1] - nums[index]) < size:
                return True
            if bucket_id + 1 in buckets and abs(buckets[bucket_id + 1] - nums[index]) < size:
                return True
            
            buckets[bucket_id] = nums[index]
            if index >= indexDiff:
                discard_bucket = self._get_bucket_id(nums[index - indexDiff], size)
                del buckets[discard_bucket]
        
        return False


"""
[ShowDuplicate]
# Part 1
# 给定一个用户观看历史的 剧集ID 列表
# print(has_duplicate_episodes([55, 66, 77, 88, 99]))  # return：False（无重复）
# print(has_duplicate_episodes([55, 66, 77, 88, 66]))  # return：True（重复：66）

# Part 2
# 判断用户是否在 K 天的时间段内，多次观看了同一集。

# Part 3
# 判断用户在过去 K 天的时间窗口内，是否观看了同一季中的至少两集。
# 若两集的 ID 之差小于等于 T，则视为属于同一季。
"""
# part 2
def check_duplicate_within_k(shows, k):
	window = set()
	low, high = 0, 0

	while high < len(shows):
		cur_show = shows[high]
		if cur_show in window:
			return True
		window.add(cur_show)

		if len(window) >= k:
			window.discard(shows[low])
			low += 1

		high += 1


	return False

# part 3
"""
We need to quickly find elements within window k that might have diff < value_diff
to do this, we can put elements into bucket where each bucket size as value_diff
so to find if there are others have value_diff we check its own and neighbor buckets
we can say each bucket only has 1 element. since if there are 2 element in 1 bucket,
meaning we find the pair

bucket size [0, t], [t + 1, 2t + 1] ...
size is t + 1
bucket index = value // size (if value = t, bucket index = 0)

loop through the elements
1. get the bucket id (ele // bucket_size)
2. if bucket_id in current buckets, we find the pair
3. check neighbor buckets
4. if none found, put into bucket
5. maintain window size, if window_size >= k (keep window_size = k - 1), remove nums[i - k]
"""
def check_show_same_season(shows, k, t):
	# bucket size t -> [0, t][t + 1, 2t + 1]...
	window_buckets = {}
	bucket_size = t + 1

	low, high = 0, 0
	while high < len(shows):
		cur_show = shows[high]
		bucket_id = cur_show // bucket_size

		if bucket_id in window_buckets:
			return True
		elif (bucket_id - 1) in window_buckets and abs(window_buckets[bucket_id - 1] - cur_show) <= t:
			return True
		elif (bucket_id + 1) in window_buckets and abs(window_buckets[bucket_id + 1] - cur_show) <= t:
			return True

		window_buckets[bucket_id] = cur_show

		while len(window_buckets) >= k:
			low_bucket_id = shows[low] // bucket_size
			del window_buckets[low_bucket_id]
			low += 1

		high += 1

	return False

# 13 Number Pairs That Match Target
# [FindPairs]
"""
Use a counter as map to record nums
For each num in counter
check target - num, num - target, target // num, num // target 
in the map, if it is, and not the same num with same index, we find a pair
Time complexity: O(N)
"""

from collections import Counter

def is_valid(counts, a, b):
	if b not in counts:
		return False
	if a != b:
		return True
	return counts[b] >= 2

def findPairs(nums: list[int], target: int) -> list[str]:
	"""
	Find all pairs of numbers where an arithmetic operation produces the target.

	Args:
		nums: List of integers
		target: Target value to match

	Returns:
		List of expression strings like ["2+4", "3*2"]
	"""
	counts = Counter(nums)

	result = set()

	for a in counts:
		# plus
		b = target - a
		if is_valid(counts, a, b):
			result.add(f'{a}+{b}')
		
		# subtract
		b = a - target
		if is_valid(counts, a, b):
			result.add(f'{a}-{b}')
		# multiply
		# check if a is 0 and target also 0
		if a == 0 and target == 0:
			for b in counts:
				if is_valid(counts, a, b):
					result.add(f'{a}*{b}')
		elif target % a == 0:
			b = target // a
			if is_valid(counts, a, b):
				result.add(f'{a}*{b}')

		# divide
		if a == 0 and target == 0:
			for b in counts:
				if b != 0 and is_valid(counts, a, b):
					result.add(f'{a}/{b}')
		elif a % target == 0:
			b = a // target
			if is_valid(counts, a, b):
				result.add(f'{a}/{b}')
	
	return list(result)

# Follow-up find 3 numbers in tuple instead of 2

"""
Calculate 2 number on operations as pairs first
Then use the 3rd number with the result of first 2 as pairs again 
"""
from collections import defaultdict


def _append_valid_combination(idx3, k, result, expr_list, op, reverse=False):
	for expr, idx1, idx2 in expr_list:
		if idx1 != idx3 and idx2 != idx3:
			final_expr = f'{k}{op}({expr})' if not reverse else f'({expr}){op}{k}'
			result.add(final_expr)


def find_triple(nums, target):
	result = set()
	pair_map = defaultdict(list)

	ops = ['+', '-', '*', '/']

	for idx1 in range(len(nums)):
		for idx2 in range(len(nums)):
			if idx1 == idx2:
				continue

			for op in ops:
				if op == '+':
					op_res = nums[idx1] + nums[idx2]
					expr = f'{nums[idx1]}+{nums[idx2]}'
					pair_map[op_res].append((expr, idx1, idx2))
				elif op == '-':
					op_res = nums[idx1] - nums[idx2]
					expr = f'{nums[idx1]}-{nums[idx2]}'
					pair_map[op_res].append((expr, idx1, idx2))
				elif op == '*':
					op_res = nums[idx1] * nums[idx2]
					expr = f'{nums[idx1]}*{nums[idx2]}'
					pair_map[op_res].append((expr, idx1, idx2))
				elif nums[idx1] % nums[idx2] == 0:
					op_res = nums[idx1] // nums[idx2]
					expr = f'{nums[idx1]}/{nums[idx2]}'
					pair_map[op_res].append((expr, idx1, idx2))

	for idx3, k in enumerate(nums):
		b = target - k
		if b in pair_map:
			_append_valid_combination(idx3, k, result, pair_map[b], '+')
			_append_valid_combination(idx3, k, result, pair_map[b], '+', True)

		b = k - target
		if b in pair_map:
			_append_valid_combination(idx3, k, result, pair_map[b], '-')

		b = target + k
		if b in pair_map:
			_append_valid_combination(idx3, k, result, pair_map[b], '-', True)

		if k == 0 and target == 0:
			for _, expr_list in pair_map.items():
				_append_valid_combination(idx3, k, result, expr_list, '*')
				_append_valid_combination(idx3, k, result, expr_list, '*', True)
		elif target % k == 0:
			b = target // k
			if b in pair_map:
				_append_valid_combination(idx3, k, result, pair_map[b], '*')
				_append_valid_combination(idx3, k, result, pair_map[b], '*', True)

		if k == 0 and target == 0:
			for value, expr_list in pair_map.items():
				if value != 0:
					_append_valid_combination(idx3, k, result, expr_list, '/')
		elif k % target == 0:
			b = k // target
			if b in pair_map:
				_append_valid_combination(idx3, k, result, pair_map[b], '/')

		b = k * target
		if b in pair_map:
			_append_valid_combination(idx3, k, result, pair_map[b], '/', True)

	return result

# 14 weighted cache
#[weightedCache]
"""
We need to quickly find the cur max weight and key.
Also, when an existing key with new weight, we need to update our content
If we use a heap, it is hard to update a key's weight
We can use SortedDict in python which is treeMap in java
"""
from sortedcontainers import SortedDict

class WeightedCache:
    def __init__(self, maxTotalWeight: int):
        self.max_weight = maxTotalWeight
        self.cur_weight = 0
        # sorted by weight -> set(key)
        self.weight_to_key = SortedDict()
        # key -> (value, weight)
        self.content = {}

    def get(self, key: str) -> str:
        """
        Time complexity - O(1)
        """
        if key not in self.content:
            return ""
        
        return self.content[key][0]

    def put(self, key: str, value: str, weight: int) -> None:
        """
        Time complexity - O(lgn) -> at most 1 item will be evicted
        If weight > all weights, this key will be evicted first and total weight back to normal
        If weight < largest weight, only the largest will be evicted since the loss delta make 
        total weight back to normal
        """
        if weight > self.max_weight:
            return

        # we need to update cache with new weight
        if key in self.content:
            old_weight = self.content[key][1]
            self.weight_to_key[-old_weight].discard(key)
            self.cur_weight -= old_weight

        # update content
        self.content[key] = (value, weight)
        if -weight not in self.weight_to_key:
            self.weight_to_key[-weight] = set()
        self.weight_to_key[-weight].add(key)
        self.cur_weight += weight

        # check weight
        while self.cur_weight > self.max_weight:
            largest_weight, largest_key_set = next(iter(self.weight_to_key.items()))
            largest_key = largest_key_set.pop()
            if not largest_key_set:
                del self.weight_to_key[largest_weight]

            del self.content[largest_key]
            self.cur_weight -= -largest_weight





# weighted random
#[weightedRandom]
"""
Use prefix sum, for each number, the prefix sum represent the total weights at this
point. And random number in range [1, total weight] falls into the prefix sum array
the larger the number, the larger the area it covers in this array. thus we random pick 
by its weight
"""
import random
class Solution(object):

    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.weights = []
        self.total_weight = 0
        for weight in w:
            self.total_weight += weight
            self.weights.append(self.total_weight)
        

    def pickIndex(self):
        """
        :rtype: int
        """
        target = random.randint(1, self.total_weight)
        
        left, right = 0, len(self.weights) - 1
        ans = -1
        while left <= right:
            mid = (right + left) // 2
            if self.weights[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                ans = mid
        return ans

# 12 Timer function
# [Timer]
def timer(seconds: int) -> str:
    """
    Convert seconds to human-readable time format recursively.

    Rules:
    - Don't show zero units except for seconds
    - Must be implemented recursively
    - Format: "X units, Y units, Z units"
    - Use singular form even if value is not 1 (e.g., "1 minutes")

    Time units:
    - 1 minute = 60 seconds
    - 1 hour = 60 minutes
    - 1 day = 24 hours
    - 1 week = 7 days
    - 1 month = 30 days

    Args:
        seconds: Non-negative integer representing seconds

    Returns:
        String in format "X units, Y units, ..."
    """
    units = [
        ('months', 30 * 24 * 60 * 60),
        ('weeks', 7 * 24 * 60 * 60),
        ('days', 24 * 60 * 60),
        ('hours', 60 * 60),
        ('minutes', 60),
        ('seconds', 1),
    ]
    return timer_recursive(seconds, units)

def timer_recursive(seconds, units):
    if not units:
        return ""

    cur_unit_name, cur_unit_seconds = units[0]
    count = seconds // cur_unit_seconds
    remainder = seconds % cur_unit_seconds

    rest = timer_recursive(remainder, units[1:])

    if cur_unit_name == 'seconds':
        return f'{count} seconds'

    if count > 0:
        result = f'{count} {cur_unit_name}'
        if rest:
            result = result + ", " + rest
        return result
    else:
        return rest


def timer_iterative(seconds):
    units = [
        ('months', 30 * 24 * 60 * 60),
        ('weeks', 7 * 24 * 60 * 60),
        ('days', 24 * 60 * 60),
        ('hours', 60 * 60),
        ('minutes', 60),
        ('seconds', 1),
    ]

    parts = []
    cur_seconds = seconds
    for unit_name, unit_seconds in units:
        count = cur_seconds // unit_seconds
        cur_seconds = cur_seconds % unit_seconds

        if count > 0 or unit_name == 'seconds':
            parts.append(f'{count} {unit_name}')
    
    return ', '.join(parts)

# 16 sliding windows
# error rate monitor

# longest consecutive same show
# [LongestCommonShow]
def longest_consecutive_same_show(shows):
	if not shows:
		return None

	low = 0
	high = 0
	max_len = 0
	cur_show = shows[low]

	while high < len(shows):
		if shows[high] != cur_show:
			cur_show = shows[high]
			low = high
		
		max_len = max(max_len, high - low + 1)
		high += 1

	return max_len

# longest consecutive unique show
# [LongestUniqueShow]
class Solution:
    """
    Use a window to record current substring
    the window maps char to their index
    2 pointers low and high as edge of window
    if high encounter a char already in the window
    move low forward to the window[s[high]] + 1
    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        # a window with key - char, value - char's index
        window_to_index = {}
        low = 0
        high = 0

        max_len = 0

        while high < len(s):
            c = s[high]
            if c in window_to_index:
                dup_index = window_to_index[c]
                # move low to dup_index + 1
                while low <= dup_index:
                    del window_to_index[s[low]]
                    low += 1
            
            window_to_index[c] = high
            max_len = max(max_len, high - low + 1)
            high += 1
        
        return max_len

# 16.5 Longest Substring Without Repeating Characters
# 第一部分很简单，不太记得细节了
# 第二部分是一道和力扣 伞 很像的用sliding window做的题目
# 第三部分：find the number of unique pairs of strings from a list such that the two strings in a pair share no common characters
# Example:
# input: {“apple”, “banana”, “peach”, “kiwi”}
# unique pairs:  [apple, kiwi], [peaches, kiwi], [banana, kiwi]
# 面试官要求用O(nlogn) 的方法做, 我只想出来brute force的

#[UniquePairs]
"""
When compare 2 strings, we can use 2 sets to represet them and
do a n^2 loop. If average length of string is m
the time complexity could be O(n^2 * m) -> since each comparison takes m

I think we only care about whether a char is shown in another string or not. 
We can use bitmask to represent a string to simplify the comparison. 
And each bit is 1 if the char at that position is present.
So aabd could be 1011. And we can AND 2 masks if result is 0 then no common characters

Time complexity: O(n*m + n^2) - m average length of string
"""

from collections import Counter

def _convert_to_mask(show):
	mask = 1
	for c in show:
		mask += 1 << (ord(c) - ord('a'))

	return mask

def find_unique_pair_count(shows):
	if not shows:
		return 0

	masks_freq = Counter([_convert_to_mask(s) for s in shows])

	result = 0

	masks = list(masks_freq.keys())
	for idx1 in range(len(masks)):
		for idx2 in range(idx1 + 1, len(masks)):
			m1 = masks[idx1]
			m2 = masks[idx2]
			if m1 & m2 == 0:
				result += masks_freq[m1] * masks_freq[m2]

	return result

"""
if a string has no common chars with another string
then mask1 & mask2 = 0
and mask2 is a submasks of (max mask (all 1s) ^ mask1)

and if we know the max mask, then we can count number of 
submasks for each masks. And check max_mask ^ mask to calculate
unique counter of pair
"""
from collections import defaultdict

def find_unique_pair_count_with_26_alphabets(shows):
	size = 26
	MAX_MASK = (1 << size) - 1

	masks = set([_convert_to_mask(s) for s in shows])
	submasks = defaultdict(int)

	for counter_mask in range(MAX_MASK + 1):
		for mask in masks:
			if counter_mask & mask > 0:
				submasks[counter_mask] += 1
			# print(counter_mask)

	result = 0
	for mask in masks:
		counter_mask = MAX_MASK ^ mask
		if submasks[counter_mask] > 0:
			result += submasks[counter_mask]

	# each pari counted twice
	return result // 2


# 17 topo sorts
"""
Topological sort
1 map as graph for edges
1 map as indegree, a node can be visited when ingress[node] = 0

To traverse the graph, use a queue
"""

# course schedule I
# [CourseFinish]

from collections import defaultdict
from queue import deque

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(set)
        indegree = defaultdict(int)

        for pre in prerequisites:
            first = pre[1]
            later = pre[0]

            graph[pre[1]].add(pre[0])
            indegree[pre[0]] += 1

        visited = 0
        queue = deque()
        for c in range(0, numCourses):
            if c not in indegree or indegree[c] == 0:
                queue.append(c)

        while queue:
            cur_c = queue.popleft()
            visited += 1
            if cur_c not in graph:
                continue
            
            for next_c in graph[cur_c]:
                indegree[next_c] -= 1
                if indegree[next_c] == 0:
                    queue.append(next_c)
        
        return visited == numCourses

# course schedule II
# [CourseOrder]
from collections import defaultdict
from queue import deque

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(set)
        indegree = defaultdict(int)

        for pre in prerequisites:
            first = pre[1]
            later = pre[0]

            graph[first].add(later)
            indegree[later] += 1

        result = []
        queue = deque()
        for c in range(0, numCourses):
            if c not in indegree or indegree[c] == 0:
                queue.append(c)
        
        while queue:
            cur_c = queue.popleft()
            result.append(cur_c)

            if cur_c not in graph:
                continue
            for next_c in graph[cur_c]:
                indegree[next_c] -= 1
                if indegree[next_c] == 0:
                    queue.append(next_c)
        
        if len(result) != numCourses:
            return []
        return result

# course schedule III
# course with time and ddl [courseDDL]
"""
keep a time to track current time, the smaller the time the more course we can take
when we encounter a course, if time + course_time <= deadline 
we can take it. If not, we should try to see if any course we take
takes more than course_time. If there is any, we can swap it to 
keep time smaller.

In order to do this, we use a max-heap to track the course_time we take
Time compleixty - O(nlgn) sorted nlgn, loop through with pq nlgn
"""

from queue import PriorityQueue

class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        max_heap = PriorityQueue()
        cur_time = 0

        # sorted by deadline
        sorted_courses = sorted(courses, key=lambda x: x[1])

        for course in sorted_courses:
            c_time = course[0]
            ddl = course[1]

            if cur_time + c_time <= ddl:
                # python pq is a min-heap
                max_heap.put(-c_time)
                cur_time += c_time
            elif not max_heap.empty() and -max_heap.queue[0] > c_time:
                larger_c_time = -max_heap.get()
                max_heap.put(-c_time)
                cur_time += c_time - larger_c_time

        return max_heap.qsize()


# 13 command undo
# [CommandUndo]
from collections import defaultdict

class HistoryNode:
	def __init__(self, cmd, tags):
		self.cmd = cmd
		self.tags = tags
		self.alive = True

class CommandSystem:
	def __init__(self):
		self.history = []
		# tag -> stack of nodes
		self.tag_history = defaultdict(list)

	def execute(self, cmd, tags):
		node = HistoryNode(cmd, tags)
		self.history.append(node)

		for tag in tags:
			self.tag_history[tag].append(node)

	def _pop_stack(self, stack):
		while stack and not stack[-1].alive:
			stack.pop()

		if not stack:
			return None
		
		last_node = stack.pop()
		last_node.alive = False

		return last_node

	def undo(self, tag=None):
		if tag is None:
			last_node = self._pop_stack(self.history)

			return f'undo {last_node.cmd}' if last_node else None

		if tag not in self.tag_history or not self.tag_history[tag]:
			return None

		# pop deleted node first
		tag_stack = self.tag_history[tag]
		last_node = self._pop_stack(tag_stack)

		return f'undo {last_node.cmd}' if last_node else None




"""
Brutal force, keep a single list, and trace back by tag if needed. Undo can be O(N)

To undo and find the right tag stack fast without tracing back
We keep a global history as linkedlist
Each tag history is also a linkedlist
so we can easily remove a node with O(1)

To quickly find related nodes for a cmd, Each node in tag has a pointer to global history node
Each node in global history has pointers to tags history nodes

Thus we can quickly find all related nodes for a tag or cmd and remove them from history

"""
from collections import defaultdict

class HistoryNode:
	def __init__(self, cmd, tags):
		self.prev = None
		self.next = None
		self.cmd = cmd
		self.tags = tags

		self.linked_nodes = []

	def eject_from_list(self):
		if self.prev:
			self.prev.next = self.next
		if self.next:
			self.next.prev = self.prev

		self.next = None
		self.prev = None


class LinkedList:
	def __init__(self):
		self.head = HistoryNode("", None)
		self.tail = HistoryNode("", None)

		self.head.next = self.tail
		self.tail.prev = self.head

	# append to tail
	def add_node(self, node):
		actual_tail = self.tail.prev

		node.prev = actual_tail
		node.next = self.tail

		actual_tail.next = node
		self.tail.prev = node

	def get_last(self):
		last_node = self.tail.prev

		if last_node != self.head:
			return last_node
		return None

class CommandSystemAllLinked:
	"""
	Each tag history is also a linkedlist
	Each node in tag has a pointer to global history node
	Each node in global history has pointers to tags history nodes

	Time comlexity:
	- execute: O(k) - k tag size
	- undo: O(k) - k tag size
	"""
	def __init__(self):
		self.global_history = LinkedList()
		# tag -> LinkedList
		self.tag_history = {}

	def execute(self, cmd, tags):
		history_node = HistoryNode(cmd, tags)
		tag_nodes = {tag: HistoryNode(cmd, tags) for tag in tags}

		self.global_history.add_node(history_node)
		for tag in tags:
			if tag not in self.tag_history:
				self.tag_history[tag] = LinkedList()
			self.tag_history[tag].add_node(tag_nodes[tag])
			# link tag node to global node
			tag_nodes[tag].linked_nodes.append(history_node)

		# link the history_node
		history_node.linked_nodes += list(tag_nodes.values())

	def undo(self, tag=None):
		if tag is None:
			last_global_node = self.global_history.get_last()
			if not last_global_node:
				return None

			last_global_node.eject_from_list()
			for tag_node in last_global_node.linked_nodes:
				tag_node.eject_from_list()

			return f'undo {last_global_node.cmd}'
		else:
			if tag not in self.tag_history:
				return None

			tag_last_node = self.tag_history[tag].get_last()
			if not tag_last_node:
				return None

			global_history_node = tag_last_node.linked_nodes[0]
			all_linked_tag_nodes = global_history_node.linked_nodes

			global_history_node.eject_from_list()
			for node in all_linked_tag_nodes:
				node.eject_from_list()

			return f'undo {tag_last_node.cmd}'

# FILO test with no tag undo
sys = CommandSystemAllLinked()
sys.execute("A", {"x"})
sys.execute("B", {"y"})

assert sys.undo() == "undo B"
assert sys.undo() == "undo A"
assert sys.undo() is None


# test undo with tag
sys = CommandSystemAllLinked()
sys.execute("A", {"x"})
sys.execute("B", {"y"})
sys.execute("C", {"x", "z"})

assert sys.undo("x") == "undo C"   # most recent command with tag x
assert sys.undo("x") == "undo A"
assert sys.undo("x") is None

# test undo skip latest cmd
sys = CommandSystemAllLinked()
sys.execute("A", {"x"})
sys.execute("B", {"y"})
sys.execute("C", {"z"})

assert sys.undo("x") == "undo A"
assert sys.undo() == "undo C"
assert sys.undo() == "undo B"
assert sys.undo() is None

# test undo multi tags
sys = CommandSystemAllLinked()
sys.execute("A", {"x", "y"})
sys.execute("B", {"x", "y"})

assert sys.undo("x") == "undo B"
assert sys.undo("y") == "undo A"
assert sys.undo() is None


# Mixed tags and plain undo order
sys = CommandSystemAllLinked()
sys.execute("A", {"x"})
sys.execute("B", {"y"})
sys.execute("C", {"x"})
sys.execute("D", {"z"})

assert sys.undo("x") == "undo C"   # removes C from middle
assert sys.undo() == "undo D"	  # now D is latest remaining
assert sys.undo("y") == "undo B"
assert sys.undo() == "undo A"
assert sys.undo() is None



# 18 tag co-occurence tracker 
# [TagTracker]
"""
Is the tag group in record interaction always size > 2?
is the timestamp always arrives in ascending order?
What if queried co-tag not in history

Use a map record tag pairs -> timestamps
when query a tag pair, binary search on the timestamp history for counting
"""

"Clarify Questions 0 - will timestamp arrives always in ascending order?"
"Clarify Questions 1 - will the record_interaction take less than 2 tags as parameter"
"Clarify Questions 2 - will the get_co_occurance_count take 2 same tags as parameter"
"Clarify Questions 3 - will there be a range of valid_after"
"Clarify Questions 4 - what if queried tag not in history?"

"""
Example
tag_analytics.record_interaction(['t1', 't2'], 0)
tag_analytics.record_interaction(['t1', 't3'], 20)
tag_analytics.record_interaction(['t1', 't2', 't3'], 40)

tag_analytics.get_co_occurrence_count('t1', 't2', 0) -> 2
tag_analytics.get_co_occurrence_count('t1', 't3', 30) -> 1
tag_analytics.get_co_occurrence_count('t1', 't3', 50) -> 0
tag_analytics.get_co_occurrence_count('t1', 't4', 50) -> None ?

tag_analytics.get_most_frequent_tag('t2', 0) -> 't1', 2
"""


from itertools import combinations
from collections import defaultdict
from bisect import bisect_left

class CoTagSearch:
	def __init__(self):
		self.tag_pair_history = defaultdict(list)
		self.tag_neighbors = defaultdict(set)

	def _get_sorted_tag_pair(self, tag1, tag2):
		return (tag1, tag2) if tag1 < tag2 else (tag2, tag1)

	# O(n^2)
	def record_interaction(self, tags, timestamp):
		if len(tags) < 2:
			raise ValueError('')

		# combination calculation for C(n, 2) is n*(n-1) ~ n^2 
		for tag_pair in combinations(tags, 2):
			tag1, tag2 = tag_pair[0], tag_pair[1]
			sorted_tag_pair = self._get_sorted_tag_pair(tag1, tag2)
			self.tag_pair_history[sorted_tag_pair].append(timestamp)

			self.tag_neighbors[sorted_tag_pair[0]].add(sorted_tag_pair[1])
			self.tag_neighbors[sorted_tag_pair[1]].add(sorted_tag_pair[0])

	# O(lgN)
	def get_co_occurrence_count(self, tag1, tag2, valid_after):
		sorted_tag_pair = self._get_sorted_tag_pair(tag1, tag2)

		if sorted_tag_pair not in self.tag_pair_history:
			return None

		history = self.tag_pair_history[sorted_tag_pair]

		idx = bisect_left(history, valid_after)

		return len(history) - idx

	# k-neighbors, n-history length | O(k*lgN)
	def get_most_frequenct_pair(self, tag, valid_after):
		if tag not in self.tag_neighbors:
			return None

		max_count = -1
		max_tag = None

		for other in self.tag_neighbors[tag]:
			count = self.get_co_occurrence_count(tag, other, valid_after)
			if count > max_count:
				max_count = count
				max_tag = other

		return (max_tag, max_count)

	def get_top_k_frequent_pairs(self, tag, valid_after, k):
		from queue import PriorityQueue


		if tag not in self.tag_neighbors:
			return None

		# item - (count, other_tag)
		pq = PriorityQueue()

		for other in self.tag_neighbors[tag]:
			count = self.get_co_occurrence_count(tag, other, valid_after)

			if pq.qsize() == k and pq.queue[0][0] < count:
				pq.get()
				pq.put((count, other))
			elif pq.qsize() < k:
				pq.put((count, other))

		result = []
		while not pq.empty():
			result.append(pq.get())

		return result



	def _binary_search(self, history, target):
		"""
		We have duplicates in the history, thus we are finding the left most position
		"""
		left, right = 0, len(history) - 1
		ans = -1

		while left <= right:
			mid = (left + right) // 2
			if history[mid] >= target:
				right = mid - 1
				ans = mid
			else:
				left = mid + 1

		return ans

	def _binary_search_exclude(history, target):
		"""
		If we exclude the target, we are finding the left most index where arr[index] > target
		"""
		left, right = 0, len(history)

		while left < right:
			mid = (left + right) // 2
			if history[mid] <= target:
				left = mid + 1
			else:
				right = mid

		return left


analytics = TagAnalytics()

analytics.record_interaction(['robotics', 'ai'], 100)
analytics.record_interaction(['ai', 'robotics'], 100)
# analytics.record_interaction(['ai', 'k'], 100)
# analytics.record_interaction(['ai', 's'], 90)
analytics.record_interaction(['ai', 'ethics'], 120)
analytics.record_interaction(['ai', 'robotics'], 130)
analytics.record_interaction(['ai', 'ethics', 'robotics'], 140)

print(analytics.tag_pair_history)

assert analytics.get_co_occurrence_count('robotics', 'ai', 110) == 2
assert analytics.get_co_occurrence_count('robotics', 'ai', 100) == 4
assert analytics.get_co_occurrence_count('ai', 'ethics', 100) == 2
assert analytics.get_co_occurrence_count('ai', 'ethics', 150) == 0
assert analytics.get_co_occurrence_count('ai', 'ethics', 10) == 2

assert analytics.get_most_frequenct_pair('ai', 100) == ('robotics', 4)
assert analytics.get_most_frequenct_pair('ethics', 100) == ('ai', 2)

print(analytics.get_top_k_frequent_pairs('ai', 100, 2))
assert set(analytics.get_top_k_frequent_pairs('ai', 100, 2)) == set([(4, 'robotics'), (2, 'ethics')])




import uuid

class MultiTagSearch:
	def __init__(self):
		# tag -> history as (timestamp, post_id)
		self.tag_history = defaultdict(list)

	def record_interaction(self, tags, timestamp):
		post_id = uuid.uuid4()
		for tag in tags:
			self.tag_history[tag].append((timestamp, post_id))

	def get_multi_occurance_count(self, tags, valid_after):
		histories = []
		for tag in tags:
			if tag in self.tag_history:
				history = self.tag_history[tag]

				history.sort(key=lambda x: x[0])

				valid_index = self._binary_search(history, valid_after)
				if valid_index >= 0 and len(history) - valid_index > 0:
					histories.append(set(history[valid_index:]))

		if not histories:
			return 0
		
		final_set = histories[0]
		for history in histories[1:]:
			final_set.intersection_update(history)

		return len(final_set)

	def _binary_search(self, history, target):
		"""
		We have duplicates in the history, thus we are finding the left most position
		"""
		left, right = 0, len(history) - 1
		ans = -1

		while left <= right:
			mid = (left + right) // 2
			if history[mid] >= target:
				right = mid - 1
				ans = mid
			else:
				left = mid + 1

		return ans


# 19 find execution time of a dag
# [DAGTime]

"""
Topological sort the graph and find the entry nodes
For the DAG, the min execution time is determined by the
longest task time node. For each node, the process time of it
from starting time is max(dependecies_process_time) + task_time

So we could use a map record nodes max_process_time.
For any node max_process_time it is
max([max_process_time of dependency nodes]) + task_time

We iterat the whole graph once and we can find the process time
Also need to detect cycle -> count processed node, it
count != total node then failed to process whole graph

Time Complexity - O(E + N) - E count of edges | N - count of nodes
"""

"Clarify Question 0 - input format, task times map and dependencies as tuples"
"Clarify Question 1 - can there be cycle in the input"
"Clarify Question 2 - you mentioned there will machines process task, are we limited on machines?"

"""
Example
A - B - D -F
C -/    \\-E

A - 3
C - 1
B - 2
D - 2
F - 1
E - 3

minimum time - A - B - D - E -> 10
"""

from collections import deque, defaultdict

class DAGProcessor:
	# task times: key-take name, value-process time
	# dependencies: (U, V), to process V, U must be processed first
	def minimum_process_time(self, task_times: dict[int], dependencies: list[tuple[str, str]]):
		graph = {task: [] for task in task_times.keys()}
		indegree = {task: 0 for task in task_times.keys()}
		max_task_time = defaultdict(int)
		visited = 0

		for u, v in dependencies:
			indegree[v] += 1
			graph[u].append(v)

		queue = deque()
		for t in indegree.keys():
			if indegree[t] == 0:
				queue.append(t)
				max_task_time[t] = task_times[t]

		max_process_time = 0
		while queue:
			cur_t = queue.popleft()
			max_process_time = max(max_process_time, max_task_time[cur_t])
			visited += 1

			for next_t in graph[cur_t]:
				# max(dependecies_process_time) + task_time
				max_task_time[next_t] = max(max_task_time[next_t], max_task_time[cur_t] + task_times[next_t])

				indegree[next_t] -= 1
				if indegree[next_t] == 0:
					queue.append(next_t)

		if visited != len(task_times):
			return None

		return max_process_time

	def minimum_process_time_with_schedule(self, task_times: dict[int], dependencies: list[tuple[str, str]]):
		graph = {task: [] for task in task_times.keys()}
		indegree = {task: 0 for task in task_times.keys()}
		max_task_time = defaultdict(int)
		visited = 0

		# start-time = max(end time of dependencies)
		# end-time = start-time + task_times[cur_t] 
		task_schedule = defaultdict(tuple)

		for u, v in dependencies:
			indegree[v] += 1
			graph[u].append(v)

		queue = deque()
		for t in indegree.keys():
			if indegree[t] == 0:
				queue.append(t)
				max_task_time[t] = task_times[t]
				task_schedule[t] = (0, task_times[t])

		max_process_time = 0
		while queue:
			cur_t = queue.popleft()
			max_process_time = max(max_process_time, max_task_time[cur_t])
			visited += 1

			for next_t in graph[cur_t]:
				if max_task_time[cur_t] + task_times[next_t] > max_task_time[next_t]:
					max_task_time[next_t] = max_task_time[cur_t] + task_times[next_t]
					task_schedule[next_t] = (max_task_time[cur_t], max_task_time[next_t])

				indegree[next_t] -= 1
				if indegree[next_t] == 0:
					queue.append(next_t)

		if visited != len(task_times):
			raise ValueError('detect cycle in DAG')

		return max_process_time, task_schedule

	def minmum_machine_needed(self, schedule):
		from queue import PriorityQueue

		sorted_schedule = sorted(schedule, key=lambda x: (x[0], x[1]))
		# store end time
		pq = PriorityQueue()

		machine_count = 0
		for sch in schedule:
			start, end = sch[0], sch[1]
			while not pq.empty() and pq.queue[0] <= start:
				pq.get()

			pq.put(end)
			machine_count = max(machine_count, pq.qsize())

		return machine_count


	def get_max_path(self, task_times: dict[int], dependencies: list[tuple[str, str]]):
		graph = {task: [] for task in task_times.keys()}
		indegree = {task: 0 for task in task_times.keys()}
		max_task_time = defaultdict(int)
		visited = 0

		# key - node name, value - parent node with max process time amoung all parents
		max_process_parent = {}

		for u, v in dependencies:
			indegree[v] += 1
			graph[u].append(v)

		queue = deque()
		for t in indegree.keys():
			if indegree[t] == 0:
				queue.append(t)
				max_task_time[t] = task_times[t]

		max_process_time = 0
		max_process_node = None
		while queue:
			cur_t = queue.popleft()
			visited += 1

			if max_task_time[cur_t] > max_process_time:
				max_process_time = max_task_time[cur_t]
				max_process_node = cur_t

			for next_t in graph[cur_t]:
				if max_task_time[cur_t] + task_times[next_t] > max_task_time[next_t]:
					max_task_time[next_t] = max_task_time[cur_t] + task_times[next_t]
					max_process_parent[next_t] = cur_t

				indegree[next_t] -= 1
				if indegree[next_t] == 0:
					queue.append(next_t)

		if visited != len(task_times):
			return None

		# graph is empty
		if not max_process_node:
			return (0, None)

		path = []
		cur = max_process_node
		while cur is not None:
			path.append(cur)
			cur = max_process_parent.get(cur, None)

		path.reverse()

		return (max_process_time, path)


task_times_example = {'A': 3, 'B': 2, 'C': 1, 'D': 2, 'E': 3, 'F': 1}
dependencies_example = [('A', 'B'), ('C', 'B'), ('B', 'D'), ('D', 'F'), ('D', 'E')]

task_times = {'A': 2, 'B': 2, 'C': 1, 'D': 2, 'E': 3}
dependencies = {('A', 'B'), ('A', 'C'), ('B', 'E'), ('C', 'E'), ('C', 'D')}
dependencies_2 = {('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')}
dependencies_3 = {('A', 'B'), ('B', 'C'), ('C', 'A')}

task_times_2 = {}
dependencies_4 = {}

dp = DAGProcessor()

assert dp.minimum_process_time(task_times_example, dependencies_example) == 10

# test example
assert dp.minimum_process_time(task_times, dependencies) == 7
# test straight dag
assert dp.minimum_process_time(task_times, dependencies_2) == 10
# test no edge, only nodes
assert dp.minimum_process_time(task_times, []) == 3

# test cycle
try:
	dp.minimum_process_time(task_times, dependencies_3)
except ValueError as e:
	assert str(e) == 'detect cycle in DAG'
else:
	raise AssertionError('expecting excetpion')

# test empty input
assert dp.minimum_process_time(task_times_2, dependencies_4) == 0

assert dp.minimum_process_time_with_schedule(task_times, dependencies) == (
	7, {'A': (0, 2), 'B': (2, 4), 'C': (2, 3), 'D': (3, 5), 'E': (4, 7)}
)


assert dp.get_max_path(task_times, dependencies) == (7, ['A', 'B', 'E'])
assert dp.get_max_path(task_times, dependencies_2) == (10, ['A', 'B', 'C', 'D', 'E'])


# 20 org level tree count and imbalance
# [OrgTree]

"""
travel the tree as dfs
Time complexity: O(N) - visit each node once
"""

"Clarify Question 0 - input format, a tree root node"
"Clarify Question 1 - are names unique?"
"Clarify Question 2 - if an employee has no report, no matter what title is, consider as IC - as level is 1?"
"Clarify Question 3 - if an employee has only 1 report, the level diff is always 0 right? - if T is < 0 then it is imbalanced"
"Clarify Question 4 - if an employee has only 0 report, it is not considered imbalanced right?"
"Clarify Question 5 - will the input guarantee to be a tree? will the input tree has cycle?"


"""
Example

    A (VP)
   /      \\ 
 B (eng)   C (manager)
           \\
           D (engineer)

levels
A - 3
B - 1
C - 2
D - 1

"""

from collections import defaultdict

class EmployeeNode:
	def __init__(self, name, title, reports):
		self.name = name
		self.title = title
		self.reports = reports


class OrgLevelProcessor:
	# return org level of root employee, level map historgram, imbalanced employee
	def process_org_level(self, org_root, T):
		# title -> level -> count
		histogram = defaultdict(lambda: defaultdict(int))
		imbalanced = set()

		highest_level = self._traverse_tree(org_root, T, histogram, imbalanced)

		return (highest_level, histogram, imbalanced)

	def _traverse_tree(self, node, T, histogram, imbalanced):
		if not node.reports:
			histogram[node.title][1] += 1
			return 1

		report_levels = []
		for report in node.reports:
			report_levels.append(self._traverse_tree(report, T, histogram, imbalanced))

		max_report_lvl = max(report_levels)
		min_report_lvl = min(report_levels)

		if max_report_lvl - min_report_lvl > T:
			imbalanced.add(node.name)

		level = max_report_lvl + 1
		histogram[node.title][level] += 1

		return level

processor = OrgLevelProcessor()

# test example
node1 = EmployeeNode('Alice', 'VP', [])
node2 = EmployeeNode('Bob', 'Engineer', [])
node3 = EmployeeNode('Carol', 'Manager', [])
node4 = EmployeeNode('Dave', 'Engineer', [])

node1.reports = [node2, node3]
node3.reports = [node4]

root_level, histogram, imbalanced = processor.process_org_level(node1, 0)

assert root_level == 3
assert histogram == {'Engineer': {1: 2}, 'Manager': {2: 1}, 'VP': {3: 1}}
assert imbalanced == set(['Alice'])


# test single node tree
node_single = EmployeeNode('Alice', 'VP', [])

root_level, histogram, imbalanced = processor.process_org_level(node_single, 0)

assert root_level == 1
assert histogram == {'VP': {1: 1}}
assert imbalanced == set()

# test single chain tree
node1 = EmployeeNode('Alice', 'VP', [])
node2 = EmployeeNode('Carol', 'Manager', [])
node3 = EmployeeNode('Dave', 'Engineer', [])

node1.reports = [node2]
node2.reports = [node3]

root_level, histogram, imbalanced = processor.process_org_level(node1, 0)

assert root_level == 3
assert histogram == {'Engineer': {1: 1}, 'Manager': {2: 1}, 'VP': {3: 1}}
assert imbalanced == set()

# test employee with different level same title
node1 = EmployeeNode('Alice', 'VP', [])
node2 = EmployeeNode('Bob', 'Engineer', [])
node3 = EmployeeNode('Carol', 'Manager', [])
node4 = EmployeeNode('Dave', 'Engineer', [])
node5 = EmployeeNode('Carol2', 'Manager', [])

node1.reports = [node2, node3]
node3.reports = [node4, node5]

root_level, histogram, imbalanced = processor.process_org_level(node1, 0)

assert root_level == 3
assert histogram == {'Engineer': {1: 2}, 'Manager': {2: 1, 1: 1}, 'VP': {3: 1}}
assert imbalanced == set(['Alice'])

# test other threshold 
node1 = EmployeeNode('Alice', 'VP', [])
node2 = EmployeeNode('Bob', 'Engineer', [])
node3 = EmployeeNode('Carol', 'Manager', [])
node4 = EmployeeNode('Dave', 'Engineer', [])
node5 = EmployeeNode('Dave2', 'Engineer', [])
node6 = EmployeeNode('Dave3', 'Engineer', [])
node7 = EmployeeNode('Dave4', 'Engineer', [])

node1.reports = [node2, node3]
node3.reports = [node4, node5]
node5.reports = [node6]
node6.reports = [node7]

root_level, histogram, imbalanced = processor.process_org_level(node1, 1)

assert root_level == 5
assert histogram == {'Engineer': {1: 3, 2: 1, 3: 1}, 'Manager': {4: 1}, 'VP': {5: 1}}
assert imbalanced == set(['Alice', 'Carol'])



class EmployeeNodeWithParent:
	def __init__(self, name, title, reports):
		self.name = name
		self.title = title
		self.reports = reports

		self.parent = None
		self.level = None

class OrgLevelTreeCountFollowup:
	def org_tree_summarize_with_cycle(self, org_root, T):
		# title -> level -> count
		histogram = defaultdict(lambda: defaultdict(int))
		imbalanced = set()
		visited = set()

		highest_level = self._traverse_tree_with_cycle_detection(org_root, T, histogram, imbalanced, visited)

		return (highest_level, histogram, imbalanced)

	def _traverse_tree_with_cycle_detection(self, node, T, histogram, imbalanced, visited):
		# since node.name is unique
		if node.name in visited:
			print(f'cycle detected on {node.name}')
			return None

		visited.add(node.name)

		if not node.reports:
			histogram[node.title][1] += 1
			return 1

		report_levels = []
		for report in node.reports:
			report_level = self._traverse_tree_with_cycle_detection(report, T, histogram, imbalanced, visited)
			if report_level is not None:
				report_levels.append(report_level)

		max_report_lvl = max(report_levels)
		min_report_lvl = min(report_levels)

		if max_report_lvl - min_report_lvl > T:
			imbalanced.add(node.name)

		level = max_report_lvl + 1
		histogram[node.title][level] += 1

		return level

	def org_tree_summarize_with_top_k(self, org_root, T, k):
		# title -> level -> count
		histogram = defaultdict(lambda: defaultdict(int))
		imbalanced = set()
		title_level = defaultdict(int)

		highest_level = self._traverse_tree_with_title_level_pair(org_root, T, histogram, imbalanced, title_level)

		top_k_pair = self._calculate_top_k_title_level(k, title_level)

		return (highest_level, histogram, imbalanced, top_k_pair)

	def _traverse_tree_with_title_level_pair(self, node, T, histogram, imbalanced, title_level):
		if not node.reports:
			title_level[(node.title, 1)] += 1
			histogram[node.title][1] += 1
			return 1

		report_levels = []
		for report in node.reports:
			report_levels.append(self._traverse_tree_with_title_level_pair(report, T, histogram, imbalanced, title_level))

		max_report_lvl = max(report_levels)
		min_report_lvl = min(report_levels)

		if max_report_lvl - min_report_lvl > T:
			imbalanced.add(node.name)

		level = max_report_lvl + 1
		histogram[node.title][level] += 1
		title_level[(node.title, level)] += 1

		return level

	def _calculate_top_k_title_level(self, k, title_level):
		from queue import PriorityQueue
		
		pq = PriorityQueue()

		for pair, count in title_level.items():
			if pq.qsize() < k:
				pq.put((count, pair))
			elif pq.queue[0][0] < count:
				pq.get()
				pq.put((count, pair))

		result = []
		while not pq.empty():
			result.append(pq.get()[1])

		return result


	def org_tree_summarize_with_deepest_employee(self, org_root, T):
		# title -> level -> count
		histogram = defaultdict(lambda: defaultdict(int))
		imbalanced = set()

		highest_level, deepest_employees = self._traverse_tree_with_deepest_employee(org_root, T, histogram, imbalanced)

		return (highest_level, histogram, imbalanced, deepest_employees)

	"""
	Employee with highest level as report has deepest depth employee
	"""
	def _traverse_tree_with_deepest_employee(self, node, T, histogram, imbalanced):
		if not node.reports:
			histogram[node.title][1] += 1
			return (1, [node.name])

		report_levels = []
		deepest_employees = []
		deepest_level = -1
		for report in node.reports:
			level, deepest_report_names = self._traverse_tree_with_deepest_employee(report, T, histogram, imbalanced)
			report_levels.append(level)

			if deepest_level < level:
				deepest_employees = deepest_report_names[:]
				deepest_level = level
			elif deepest_level == level:
				deepest_employees += deepest_report_names

		max_report_lvl = max(report_levels)
		min_report_lvl = min(report_levels)

		if max_report_lvl - min_report_lvl > T:
			imbalanced.add(node.name)

		level = max_report_lvl + 1
		histogram[node.title][level] += 1

		return (level, deepest_employees)

	# return org level of root employee, level map historgram, imbalanced employee
	def org_tree_summarize_with_parent_pointer(self, org_root, T):
		# title -> level -> count
		histogram = defaultdict(lambda: defaultdict(int))
		imbalanced = set()

		highest_level = self._traverse_tree_parent_pointer(org_root, T, histogram, imbalanced)

		return (highest_level, histogram, imbalanced)

	def _traverse_tree_parent_pointer(self, node, T, histogram, imbalanced):
		if not node.reports:
			histogram[node.title][1] += 1
			node.level = 1
			return 1

		report_levels = []
		for report in node.reports:
			report_levels.append(self._traverse_tree_parent_pointer(report, T, histogram, imbalanced))

		max_report_lvl = max(report_levels)
		min_report_lvl = min(report_levels)

		if max_report_lvl - min_report_lvl > T:
			imbalanced.add(node.name)

		level = max_report_lvl + 1
		histogram[node.title][level] += 1

		node.level = level

		return level

	# when attach new IC, we might need to remove cur node out of imbalanced or add it in
	# then move up to its parent to check level update
	def attach_ic_to_manager_update_results(self, T, histogram, imbalanced, ic_node, manager_node):
		# update info
		ic_node.level = 1
		histogram[ic_node.title][ic_node.level] += 1
		manager_node.reports.append(ic_node)
		ic_node.parent = manager_node

		# check imbalance and level update
		mover = manager_node
		while mover:
			report_levels = [r.level for r in mover.reports]

			max_level = max(report_levels)
			min_level = min(report_levels)

			if max_level - min_level > T and mover.name not in imbalanced:
				imbalanced.add(mover.name)
			elif max_level - min_level <= T and mover.name in imbalanced:
				imbalanced.discard(mover.name)

			# check if level need update
			if mover.level != max_level + 1:
				old_level = mover.level
				mover.level += 1
				histogram[mover.title][old_level] -= 1
				histogram[mover.title][mover.level] += 1

				if not histogram[mover.title][old_level]:
					del histogram[mover.title][old_level]

				mover = mover.parent
			else:
				# if no level change, no need to move upward
				break


# test append new IC
node1 = EmployeeNodeWithParent('Alice', 'VP', [])
node2 = EmployeeNodeWithParent('Bob', 'Engineer', [])
node3 = EmployeeNodeWithParent('Carol', 'Manager', [])
node4 = EmployeeNodeWithParent('Dave', 'Engineer', [])

node1.reports = [node2, node3]
node3.reports = [node4]

node2.parent = node1
node3.parent = node1
node4.parent = node3

root_level, histogram, imbalanced = processor.org_tree_summarize_with_parent_pointer(node1, 0)

assert root_level == 3
assert histogram == {'Engineer': {1: 2}, 'Manager': {2: 1}, 'VP': {3: 1}}
assert imbalanced == set(['Alice'])

# remove from imbalanced, reports level balanced
new_node = EmployeeNodeWithParent('Newbie', 'Engineer', [])
processor.attach_ic_to_manager_update_results(0, histogram, imbalanced, new_node, node2)

assert imbalanced == set()
assert histogram == {'Engineer': {1: 2, 2: 1}, 'Manager': {2: 1}, 'VP': {3: 1}}

# added from imbalanced
new_node2 = EmployeeNodeWithParent('Newbie2', 'Engineer', [])
processor.attach_ic_to_manager_update_results(0, histogram, imbalanced, new_node2, node3)

new_node3 = EmployeeNodeWithParent('Newbie3', 'Engineer', [])
processor.attach_ic_to_manager_update_results(0, histogram, imbalanced, new_node3, node4)

assert imbalanced == set(['Alice', 'Carol'])
assert histogram == {'Engineer': {1: 3, 2: 2}, 'Manager': {3: 1}, 'VP': {4: 1}}




# [animation]
from collections import defaultdict, deque


class AnimationComposition:
	def process_animation(self, dependencies):
		graph = defaultdict(set)
		indegree = defaultdict(int)
		nodes = set()
		visited = 0

		for dep in dependencies:
			processed_dep = dep.split(' depends on ')

			if len(processed_dep) != 2:
				raise ValueError(f'Invalid Input, dependencies cannot be processed - {dep}')

			# animation1 depends on animation2 
			animation1 = processed_dep[0]
			animation2 = processed_dep[1]

			graph[animation2].add(animation1)
			indegree[animation1] += 1

			nodes.add(animation1)
			nodes.add(animation2)


		queue = deque()
		for node in nodes:
			if indegree[node] == 0:
				queue.append(node)

		result = []
		while queue:
			cur_node = queue.popleft()
			result.append(cur_node)
			visited += 1

			for next_node in graph[cur_node]:
				indegree[next_node] -= 1
				if indegree[next_node] == 0:
					queue.append(next_node)

		if visited != len(nodes):
			raise ValueError('invalid input contains cycle')

		return result



ac = AnimationComposition()

dependencies = ['Boat depends on Ocean', 'Boat depends on Sky', 'Pratham depends on Boat', 'Mary depends on Car']



# 21 music list
# [musicList]
from sortedcontainers import SortedDict
from collections import defaultdict


class MusicPlaylist:
	def __init__(self):
		"""Initialize the playlist object."""
		self.insert_index = 0
		# sorted time -> set((insert_index, song))
		self.time_to_song = SortedDict()
		self.song_to_time = defaultdict(set)
		# (song, ts) -> insert_index
		self.song_ts_to_index = {}

	def add(self, song: str, timestamp: int) -> None:
		"""
		Time complexity: O(lgn) -> insert into SortedDict
		"""
		"""Add a record that the song was listened to at the given timestamp."""
		if (song, timestamp) in self.song_ts_to_index:
			raise ValueError('listen same song at same time')

		cur_idx = self.insert_index
		self.insert_index += 1

		if timestamp not in self.time_to_song:
			self.time_to_song[timestamp] = set()
		self.time_to_song[timestamp].add((cur_idx, song))
		self.song_to_time[song].add(timestamp)
		self.song_ts_to_index[(song, timestamp)] = cur_idx

	def getAll(self) -> list[str]:
		"""
		Time complexity: O(n*mlgm) -> n timestamps, m songs in average
		Iterator through sorted dict is O(n)
		"""
		"""
		Return all songs in timestamp order.
		If timestamps are equal, return in insertion order.
		"""
		result = []
		for timestamp, song_set in self.time_to_song.items():
			sorted_songs = [t[1] for t in sorted(list(song_set), key=lambda x: x[0])]
			result += sorted_songs

		return result

	def remove(self, song: str, timestamp: int) -> bool:
		"""
		Time complexity: O(lgn) -> delete and lookup in SortedDict is lgn
		"""
		"""
		Remove the listening record for the specific song at the specific timestamp.
		Returns True if the record existed and was removed, False otherwise.
		"""
		if (song, timestamp) not in self.song_ts_to_index:
			return False

		idx = self.song_ts_to_index[(song, timestamp)]
		del self.song_ts_to_index[(song, timestamp)]

		self.time_to_song[timestamp].discard((idx, song))
		if not self.time_to_song[timestamp]:
			del self.time_to_song[timestamp]

		self.song_to_time[song].discard(timestamp)
		if not self.song_to_time[song]:
			del self.song_to_time[song]

		return True

	def removeAll(self, song: str) -> int:
		"""
		Time complexity: O(mlgn) -> delete and lookup in SortedDict is lgn, we have m timestamps
		"""
		"""
		Remove all listening records for the given song.
		Returns the number of records removed.
		"""
		if song not in self.song_to_time:
			return 0

		timestamps = self.song_to_time[song]
		del self.song_to_time[song]

		for ts in timestamps:
			idx = self.song_ts_to_index[(song, ts)]
			del self.song_ts_to_index[(song, ts)]

			self.time_to_song[ts].discard((idx, song))
			if not self.time_to_song[ts]:
				del self.time_to_song[ts]

		return len(timestamps)







# ===================================================================================
Data Model
[DM]


seller side
Surface -> placement

						proposal -> deal -> reservation
									|
									deal_id
									|
DSP									|
advertiser/Account -> Campaign -> lineItem -> ads -> creative/media
									 |
			Targeting + bidding + frequency capping + schedule + delivery


advertiser
advertiser_id / UUID
name / str
brand / str
billing_info_id / UUID
contact / str
category / list[str]
created_at timestamp


Campaign
campaign_id / UUID
advertiser_id / UUID
total_budget / decimal
name / str
description / str
created_at / timestamp
camp_level_frequency_cap_config_id
status / enum -> [active, finished, draft]

LineItem
line_item_id / UUID
campaign_id / UUID
line_item_type / enum -> [preferred_deal, sponsorship, standard, adExchange(open auction) ...]
budget_amount / decimal
targeting_config_id / UUID
bidding_strategy_config_id  / UUID
line_item_level_frequency_cap_config_id / UUID
schedule_config_id / UUID
delivery_goal_config_id / UUID

Targeting_config
targeting_config_id
device
behavior_segments
geo
gender
age_group

bidding_strategy_config
bidding_strategy_config_id / UUID
strategy_type / enum -> [Fixed_CPM, MAX_IMP ...]
max_bid / decimal
bid_modifier_factor / json -> {sports: 0.8, shoes: 1.2}

Frequency_cap_config
frequency_cap_config_id
window_type / enum -> [Hour, Day, Rolling_day, Rolling_hour]
window_size / int
limit / int

Schedule_config
schedule_config_id
start_time / timestamp
end_time / timestamp
show_time_expression / str -> '* * * 7-9 *' everyday show ads between 7-9 am 

delivery_goal
delivery_goal_config_id
pacing_type / enum -> [even, frontloaded ...]
goal_type / enum -> [budget, imp, click]
goal_value / decimal


Ad
ad_id
creative_id
name
status / enum -> [actie, paused]

Creative
creative_id
media_asset_id
landing_page_url
type / enum -> [interactive, video, image]
review_status / enum -> [approved, denied, pending]

Media_asset
media_asset_id
asset_url
height
width

------------------------------
seller

Surface
surface_id
surface_type / enum -> [mobile, web, ctv...]
env / enum -> [Test, Prod]

Placement
placement_id
surface_id
placement_type / enum -> [banner, video-mid-roll, game ...]

Proposal
proposal_id
name
buyer_id
seller_id
status / enum -> [finished, pending, canceled]
total_budget

Deal
deal_id
proposal_id
price_cpm
total_imp
start_time
end_time
placement_type
commitment_type / enum -> [preferred_deal, standard, sponsorship]

Reservation
reservation_id
deal_id
targeting_config_id
current_delivery


















# ====================================================================================
# ====================================================================================
# ====================================================================================
Coding is actually - Coding + System
1. course schedule



1. 
Tree DFS
电面拓扑
经典拓扑排序题
店面 幺幺散留 followup是幺思酒思

是个topo sort，里扣幺幺伞流

首先是代码轮，就是普通的拓扑排序。但是输入是纯字符串所以楼主花了点额外时间提取依赖关系，导致没啥followup。




2.
command undo system
func execute(command:str, tag:list[str])
func undo(tag:optional)-> str
如果没有给tag， undo最近的command， 如果有提供tag，undo带该tag的最近的command

实现一个 class 支持 execute command,  command undo;
基本就是考察数据结构, 保留历史 execute command, 然后怎么undo last command




3.
广告组都是拓扑 轮到我来了道timed KV cache

第一轮店面 -》 实现一个可以expire的key value -〉 followup 怎么删除expired key



LC 2622. Cache With Time Limit

Follow up：clean up expired data.
对自己很生气，有个地方写法不当（逻辑没问题！！）导致test case过不了。debug 不是很友好，当场没看出问题。下来一秒解决。。。还是面试经验不够。

题就是地里那道timebased kv store，不太难，很快写完了，聊了聊内存满了怎么办，很常规。挂了挺奇怪的，可能交流不太好，面的前几家不太在状态，上来直接写，说话太少了，


非leetcode，要求实现一个可以expire的key value缓存。一开始实现了懒惰删除然后讨论一下怎么避免OOM，要求起另一个线程去删过期数据，
讨论了一下如果要求固定size怎么搞（就是LRU），最后讨论了一下API设计。

我觉得非leetcode这个说法有点误导. 不光LRU是leetcode, 可以expire的key value缓存也是leetcode: 981. Time Based Key-Value Store. 
其实这种面试考的还是leetcode, 只是稍微披了一层大家都能看穿的不重要的皮.



Design auto-expire cache，类似 LC 2622 。
面试的时候只有文字描述，需要跟面试官讨论具体实现什么，怎么实现。class AutoExpireCache:
    def __init__(self):
   
    def set(self, key, value, timestamp, expire_period):
   
    def get(self, key):
我写的是这些API。也讨论了input type是什么，cache miss的需要返回 None instead of False because False can be the value，
expire的时候是否需要return error message和error code（不需要），能否直接override key，等等。
实现比较直给，我就用了dict。需要run code，写基本的tests（assert in Python）。
Follow up 1:
如果memory不够怎么办。
我看以前的帖子里提到了cron job和lazy delete，就和面试官讨论了一下。这里没让写code。
Follow up 2:
如果memory还不够怎么办。
我感觉面试官的意思是实现sized LRU cache。class AutoExpireCache:
    def __init__(self, k):
由于一开始用的dict，面试的时候没有反应过来，后来就没写完就到时间了。其实用个OrderedDict就行了。估计gg了。





4.
一个tree related problem, 没在地里见过, 题目细节不是特别记得了，跟tree getdepth有点类似，不难，需要pass test (自己写的test case)
印度小姐姐面的我，面试中很collaborative, 据说给了我好的feedback, 感谢她



5.
店面：力扣636


6.
力扣 56


7.
店面：rate limiter coding + rate limiter system design

考了给定一个时间，比如1min，写一个print函数，input是event，有name和timestamp，要求如果在1min以内有相同的event则只print一次。

先用基本解法拿map存起来见过的event和timestamp，每次新的event进来检查上一次见这个event的时间如果间隔超过1min就输出，否则就ignore。
follow up讨论了如何删除map里过期的数据，比如用LRU，lazy delete还是cronjob定期清理。还问了如果event的timestamp是乱序的怎么维护LRU。
网飞的面试官非常看重实际情况的处理，会有很多讨论。求加米

上个月的 奈飞技术店面 问的就是地里的Log rate limiter, 先讨论下本体实现是啥样 
然后开始讨论如果数据量并特别大或者这个rate limiter要运行很久会有什么问题 
然后讨论了一下怎么删除数据 然后又实现了一下数据删除的部分 
挺实际的问题 面之前特地看了下rate limiter的system design 很有帮助


电面：一位年轻的senior 小哥，加上东欧大叔shadow. 
很free-form：implement rpc rate limiter，没有什么具体要求；有点像是把sys design 和basic coding结合。
简单实现一下后，开始讨论各种scalability concerns和quota management design。 很practical，挺有意思的。


这个题有点像lru的变种， 考同日志不输出，如果log一样的话，不要在短时间内输出(会给一个时间range)。之后会有很多follow up, 
类似于怎么删除数据，怎么处理concurrency以及有什么strategy去实现最优的删除策略




100.
parse string to integer，不能用系统自带valueOf之类的

#1 可以有-，没有+，-永远在第一位，代表负数。如果overflow，throw exception。没有special character。
#2 可以有special character，字母之类的，只要有就throw exception
#3 可以有leading zero，只有一位leading zero是valid，超过一位leading zero就要throw exception。-可以在后来任何一位，这种情况要throw exception。

记不住了， 不是ads


一轮是coding, 给一个数组，取两个数字做加减乘除操作，输出加减乘除操作后正好是target值的所有pair。follow up是三个数字做加减乘除操作后正好是target的pairs


101.
Latency tracker class 提供两个功能：

1. 存latency samples。inputs：timestamp，latency in millisecond
2. 查询给定窗口内的，指定的percentile latency。input：time window，percentile。

该类需要支持concurreny。



102.
netlifx home page has a list of shelf，每个 shelf 有一堆 title。要在 viewport（每个 shelf 最多显示 X 个）里做dedupe。
垂直滚动我们先不管（vertical viewport 视为无限）。

coding 是一堆用户看电影的历史里，找出最后 k 部电影的集合相同的用户分组，follow up 改成找任意两个人最后 k 部电影里有至少 m 部重叠就算一对朋友并返回所有这样的 pair。
参考：https://www.hack2hire.com/companies/netflix/coding-questions/68e2c587dab409e303d92946/practice?questionId=68e2d042dab409e303d92952



103.
问题 1：最大连续相同 show 的长度（15 分钟、一次过）
问题 2：最长连续不重复子数组（现场最终版过测，但不够“clean”）


104.
面经没看到，难道是新题
music playlist

支持 add， remove，shuffle
bug free 一次过，要现场跑test。

歌曲有听的timestamp，add一首歌曲何时听的。
实现一个api 返回所有收听歌曲，按照时间顺序
remove一个歌曲收听记录
删除所有歌曲收听记录。
应该没有shuffle，记错了。一个月前面的。
不难，现场写都能写出来的程度



105.
地理面经题file system，非leetcode原题

具体题干非常宽泛 需要design一个resilient file system，确保如果system crash了能够恢复file内容，就一个描述没有code。
我是大致按照这道题 (https://leetcode.com/problems/snapshot-array/description/) 来做的。
followup问了很多bottle neck和进production的话要如何改的问题


大概内容就是实现一个in memory 的 文件管理系统,

支持增删改查, 以及支持 version, 难度不高


主要是一直有 follow up, 有点小 system design 的意思,

讨论在实际系统里面怎么使用,

当天出结果, 面试过程还挺好的

和leetcode in memory FS 的区别是你要搞version control，还有改动文件
leetcode版好像是append only



106.
店面面的是保证synchronization 的kv store, 感觉java直接简单用concurrent hash map + atomic operation 就行了。
但是我面试用python准备的，只能写了个简易的 fine grained lock.


107.
白人老哥 刷题网呃呃令 follow up 是O1 bucket sort 聊的还比较high

108.
利口 尔腰妻 尔腰酒 尔尔灵

相同1，2，3系列
应该是利口 尔尔令系列

coding: 耳药器，耳药酒，耳耳凌，当场好评。


109.
电面是 力扣 三 ，这道题的各种变形，核心都是这道题

coding就是leetcode 叁的变形，把number换成了show name或者要求不太一样
但道理是相通的


110.
coding sliding window


Part 1
给定一个用户观看历史的 剧集ID 列表
print(has_duplicate_episodes([55, 66, 77, 88, 99]))  # return：False（无重复）
print(has_duplicate_episodes([55, 66, 77, 88, 66]))  # return：True（重复：66）

Part 2
判断用户是否在 K 天的时间段内，多次观看了同一集。

Part 3
判断用户在过去 K 天的时间窗口内，是否观看了同一季中的至少两集。
若两集的 ID 之差小于等于 T，则视为属于同一季。

111.
OOD design + coding

设计一个framework可以execute 有dependency的task
task interface和framework 需要自己design
太久没做Java的generic implementation所以挂了。

楼主请问一下这是ads组吗
不是


112.
第一轮代码
图题 但是边有点特殊 有起点终点 然后还有时间戳
实现两个接口
1 - 添加边
2 - 给定起点 终点 深度 起始时间 结束时间 返回所有的边
在实现前要先讨论怎么实现 有哪些tradeoff 比如正常的怎么存这个图 然后假如时间戳很多有1m个 怎么存能让搜索效率变快
凌鹰来的烙印



113.
第二轮多线程
设计个计数器 实现加一和减一 还有实现个函数 多个线程可以wait这个函数 然后计数器为0的时候 这些线程全被notify
雨林来的烙印



114.
mplement a function timer that takes a single argument seconds whose unit is an integer representing the number of seconds. The function returns a string representing the time in a human readable form up to Months (assuming 30 days in a month). Examples:

        timer(55);      // returns: 55 seconds


        timer(65);      // returns: 1 minutes, 5 seconds

        timer(3805000); // returns: 1 months, 2 weeks, 56 minutes, 40 seconds


        // Requirements:// Don’t print zero units except for seconds - example:
        // Good: 5 minutes, 0 seconds
        // Good: 2 days, 3 minutes, 10 seconds
        // Bad:  2 days, 0 hour, 3 minutes, 10 seconds
        // Don’t use standard built-in language libraries (date, time, etc..)
        // Function to be recursive - starting off iterative to find a pattern is okay
这道题并不难，但是一开始没有看到要求用递归来写，所以花了不少时间写Iterative的code，写code比较烦。需要模除月，周，天，时，分，秒。
写完后，才被提醒要求用递归来改写。但思考再三，觉得用递归比较勉强，最后写了，但来不及运行，最后就挂了。如果一开始意识到用递归，会有比较多时间来思考，应该不会这样。
这样简单的题目挂了，觉得比较冤。



116.
狸扣三个月泰格题之一 銳忒厘米特


117,
实现weighted cache的get和put
初始设置total weight limit

达到limit后把weight最大的key-value pair删除
用treemap就可以保证get和put都是O(logN)  面试官表示赞赏 说大部分人用heap导致复杂度O(N)



118.
round 1: k closest distance, leetcode 原题


119.
L1: 给定一系列观看ID, 如果有重复观看，返回True，不然False e.g. [4, 5, 100, 200, 5] -> True
L2: 如果只在最近的K window内有重复观看，返回True，就是一个sliding window
L3: 给定一个ID range T，如果两个ID距离小于等于T，认为属于一个系列，返回True，比如T=4，[1, 5, 100]-> True，但是T=3的话返回False


120.
两周前面的，形式有些特别，share screen给看了个demo UI，实际就是蠡口意思刘。



121.
9 、 Movie History
设计并实现一个用于管理用户电影浏览历史记录的类。该类应支持以下三个接口：

1. 添加电影至历史记录 (Add a Movie to the History)

记录用户何时浏览了某部电影。每部电影可以使用一个唯一标识符（字符串或整数）来表示

2. 获取浏览历史记录 (Retrieve the Browsing History)

返回按浏览顺序排列的电影列表，其中最近浏览的电影排在最后

3. 清空浏览历史记录 (Clear the Browsing History)



