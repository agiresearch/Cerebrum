import unittest
from typing import Dict, Any, Optional, List

from cerebrum.memory.apis import (
    create_memory,
    get_memory,
    update_memory,
    delete_memory,
    search_memories,
    create_agentic_memory,
    MemoryResponse,
    MemoryQuery
)


class TestMemoryAPIs(unittest.TestCase):
    """
    Unit tests for the memory API functions.
    
    These tests verify that each memory operation works correctly by making
    actual API calls to the server running at localhost:8000.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_name = "test_agent"
        self.base_url = "http://localhost:8000"
        self.test_content = "Test memory content"
        self.test_metadata = {"tags": ["test", "important"], "priority": "high"}
        self.memory_id = None  # Will be populated during test_create_memory
        
        # Create test memories for search tests
        self.search_memory_ids = []

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any created memories
        if self.memory_id:
            try:
                response = delete_memory(
                    agent_name=self.agent_name,
                    memory_id=self.memory_id,
                    base_url=self.base_url
                )
                if isinstance(response, dict) and 'response' in response:
                    resp = response['response']
                    if resp.get('success', False):
                        print(f"Cleaned up memory: {self.memory_id}")
                    else:
                        print(f"Failed to clean up memory: {self.memory_id}")
                elif hasattr(response, 'success') and response.success:
                    print(f"Cleaned up memory: {self.memory_id}")
                else:
                    print(f"Failed to clean up memory: {self.memory_id}")
            except Exception as e:
                print(f"Error cleaning up memory {self.memory_id}: {e}")
        
        # Clean up search test memories
        for mid in self.search_memory_ids:
            try:
                response = delete_memory(
                    agent_name=self.agent_name,
                    memory_id=mid,
                    base_url=self.base_url
                )
                if isinstance(response, dict) and 'response' in response:
                    resp = response['response']
                    if resp.get('success', False):
                        print(f"Cleaned up search test memory: {mid}")
                    else:
                        print(f"Failed to clean up search test memory: {mid}")
                elif hasattr(response, 'success') and response.success:
                    print(f"Cleaned up search test memory: {mid}")
                else:
                    print(f"Failed to clean up search test memory: {mid}")
            except Exception as e:
                print(f"Error cleaning up search memory {mid}: {e}")

    def _get_response_data(self, response):
        """Extract the actual response data from the response object or dictionary."""
        if isinstance(response, dict) and 'response' in response:
            return response['response']
        return response

    def test_create_basic_memory(self):
        """
        Tests basic memory creation with content and metadata.
        
        API: create_memory
        Parameters:
            - agent_name: Name of the test agent
            - content: Basic test content string
            - metadata: Dictionary with tags and priority
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Memory ID generation
        """
        response = create_memory(
            agent_name=self.agent_name,
            content=self.test_content,
            metadata=self.test_metadata,
            base_url=self.base_url
        )
        
        # print("\nCreate Memory Response:")
        # print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Store memory_id for use in subsequent tests
        self.memory_id = resp_data.get('memory_id')
        print("test 1 memory_id: ", self.memory_id)
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertIsNotNone(resp_data.get('memory_id'))
        
        return response
    
    def test_create_agentic_memory_with_context(self):
        """
        Tests agentic memory creation after populating context with multiple memories.
        
        API: create_agentic_memory
        Parameters:
            - agent_name: Name of the test agent (creates 10 different agent memories first)
            - content: Test content string
            - metadata: Dictionary with tags and priority
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Memory ID generation in agentic context
        """
        for i in range(10):
            # mock db
            _ = create_memory(
                agent_name=self.agent_name+"_"+str(i),
                content=self.test_content+"test {}".format(i),
                metadata=self.test_metadata,
                base_url=self.base_url
            )

    
        response = create_agentic_memory(
            agent_name=self.agent_name,
            content=self.test_content,
            metadata=self.test_metadata,
            base_url=self.base_url
        )
        
        # print("\nCreate Memory Response:")
        # print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Store memory_id for use in subsequent tests
        self.memory_id = resp_data.get('memory_id')
        print("test 1 memory_id: ", self.memory_id)
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertIsNotNone(resp_data.get('memory_id'))
        
        return response

    def test_create_memory_minimal(self):
        """
        Tests memory creation with only required parameters (no metadata).
        
        API: create_memory
        Parameters:
            - agent_name: Name of the test agent
            - content: Simple content string
            - base_url: Server endpoint
            
        Verifies:
            - Success with minimal parameters
            - Memory ID generation without metadata
        """
        content = "Test memory without metadata"
        
        response = create_memory(
            agent_name=self.agent_name,
            content=content,
            base_url=self.base_url
        )
        
        # print("\nCreate Memory Without Metadata Response:")
        # print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Store for cleanup
        if resp_data.get('success') and resp_data.get('memory_id'):
            self.search_memory_ids.append(resp_data.get('memory_id'))
        
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertIsNotNone(resp_data.get('memory_id'))

    def test_retrieve_memory_by_id(self):
        """
        Tests memory retrieval using memory ID.
        
        API: get_memory
        Parameters:
            - agent_name: Name of the test agent
            - memory_id: ID from previously created memory
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Content matches original
            - Metadata integrity
        """
        # First create a memory if we don't have one
        if not self.memory_id:
            create_response = self.create_basic_memory()
            create_resp_data = self._get_response_data(create_response)
            self.memory_id = create_resp_data.get('memory_id')
            if not self.memory_id:
                self.fail("Failed to create test memory")
        
        response = get_memory(
            agent_name=self.agent_name,
            memory_id=self.memory_id,
            base_url=self.base_url
        )
        
        print("\nRead Memory Response:")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        print(f"  Response Data: {resp_data}")
        print(f"  Test Content: {self.test_content}")
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertEqual(resp_data.get('content'), self.test_content)
        # Check if metadata contains the expected keys
        metadata = resp_data.get('metadata', {})
        if metadata:
            for key in self.test_metadata:
                if key in metadata:
                    print(f"  Metadata key: {key}, Expected: {self.test_metadata[key]}, Actual: {metadata[key]}")
                    self.assertEqual(metadata[key], self.test_metadata[key])

    def test_update_memory_complete(self):
        """
        Tests full memory update (both content and metadata).
        
        API: update_memory
        Parameters:
            - agent_name: Name of the test agent
            - memory_id: ID of target memory
            - content: Updated content string
            - metadata: Updated metadata dictionary
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Content update
            - Metadata update
        """
        # Create a new memory specifically for this test
        create_response = create_memory(
            agent_name=self.agent_name,
            content=self.test_content,
            metadata=self.test_metadata,
            base_url=self.base_url
        )
        
        create_resp_data = self._get_response_data(create_response)
        test_memory_id = create_resp_data.get('memory_id')
        
        if not test_memory_id:
            self.fail("Failed to create test memory")
            
        print("Created new memory ID for update test:", test_memory_id)
        
        # New content and metadata for update
        new_content = "Updated test memory content"
        new_metadata = {"tags": ["test", "updated"], "priority": "medium"}
        
        response = update_memory(
            agent_name=self.agent_name,
            memory_id=test_memory_id,
            content=new_content,
            metadata=new_metadata,
            base_url=self.base_url
        )
        
        print("\nUpdate Memory Response:")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Verify the update by reading the memory again
        read_response = get_memory(
            agent_name=self.agent_name,
            memory_id=test_memory_id,
            base_url=self.base_url
        )
        
        print("\nVerify Update (Read Memory):")
        print(f"  Response: {read_response}")
        
        read_resp_data = self._get_response_data(read_response)
        
        # Cleanup
        if test_memory_id:
            try:
                delete_memory(
                    agent_name=self.agent_name,
                    memory_id=test_memory_id,
                    base_url=self.base_url
                )
                print(f"Cleaned up test memory: {test_memory_id}")
            except Exception as e:
                print(f"Error cleaning up memory {test_memory_id}: {e}")
        
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertTrue(read_resp_data.get('success'))
        self.assertEqual(read_resp_data.get('content'), new_content)
        
        # Check if metadata contains the expected keys
        metadata = read_resp_data.get('metadata', {})
        if metadata and 'tags' in metadata:
            self.assertEqual(set(metadata['tags']), set(new_metadata['tags']))

    def test_update_memory_content(self):
        """
        Tests partial memory update (content only).
        
        API: update_memory
        Parameters:
            - agent_name: Name of the test agent
            - memory_id: ID of target memory
            - content: New content string
            - metadata: None (unchanged)
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Content update while preserving metadata
        """
        # Create a new memory specifically for this test
        create_response = create_memory(
            agent_name=self.agent_name,
            content=self.test_content,
            metadata=self.test_metadata,
            base_url=self.base_url
        )
        
        create_resp_data = self._get_response_data(create_response)
        test_memory_id = create_resp_data.get('memory_id')
        
        if not test_memory_id:
            self.fail("Failed to create test memory")
            
        print("Created new memory ID for content-only update test:", test_memory_id)
        
        # New content for update
        new_content = "Content-only update test"
        
        response = update_memory(
            agent_name=self.agent_name,
            memory_id=test_memory_id,
            content=new_content,
            metadata=None,
            base_url=self.base_url
        )
        
        print("\nUpdate Memory Content Only Response:")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Verify the update by reading the memory again
        read_response = get_memory(
            agent_name=self.agent_name,
            memory_id=test_memory_id,
            base_url=self.base_url
        )
        
        print("\nVerify Content-Only Update (Read Memory):")
        print(f"  Response: {read_response}")
        
        read_resp_data = self._get_response_data(read_response)
        
        # Cleanup
        if test_memory_id:
            try:
                delete_memory(
                    agent_name=self.agent_name,
                    memory_id=test_memory_id,
                    base_url=self.base_url
                )
                print(f"Cleaned up test memory: {test_memory_id}")
            except Exception as e:
                print(f"Error cleaning up memory {test_memory_id}: {e}")
        
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertTrue(read_resp_data.get('success'))
        self.assertEqual(read_resp_data.get('content'), new_content)

    def test_update_memory_metadata(self):
        """
        Tests partial memory update (metadata only).
        
        API: update_memory
        Parameters:
            - agent_name: Name of the test agent
            - memory_id: ID of target memory
            - content: None (unchanged)
            - metadata: Updated metadata dictionary
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Metadata update while preserving content
        """
        # Create a new memory specifically for this test
        create_response = create_memory(
            agent_name=self.agent_name,
            content=self.test_content,
            metadata=self.test_metadata,
            base_url=self.base_url
        )
        
        create_resp_data = self._get_response_data(create_response)
        test_memory_id = create_resp_data.get('memory_id')
        
        if not test_memory_id:
            self.fail("Failed to create test memory")
            
        print("Created new memory ID for metadata-only update test:", test_memory_id)
        
        # New metadata for update
        new_metadata = {"tags": ["metadata-only", "test"], "priority": "high"}
        
        response = update_memory(
            agent_name=self.agent_name,
            memory_id=test_memory_id,
            content=None,
            metadata=new_metadata,
            base_url=self.base_url
        )
        
        print("\nUpdate Memory Metadata Only Response:")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Verify the update by reading the memory again
        read_response = get_memory(
            agent_name=self.agent_name,
            memory_id=test_memory_id,
            base_url=self.base_url
        )
        
        print("\nVerify Metadata-Only Update (Read Memory):")
        print(f"  Response: {read_response}")
        
        read_resp_data = self._get_response_data(read_response)
        
        # Cleanup
        if test_memory_id:
            try:
                delete_memory(
                    agent_name=self.agent_name,
                    memory_id=test_memory_id,
                    base_url=self.base_url
                )
                print(f"Cleaned up test memory: {test_memory_id}")
            except Exception as e:
                print(f"Error cleaning up memory {test_memory_id}: {e}")
        
        # Assertions
        self.assertTrue(resp_data.get('success'))
        self.assertTrue(read_resp_data.get('success'))
        
        # Check if metadata contains the expected keys
        metadata = read_resp_data.get('metadata', {})
        if metadata and 'tags' in metadata:
            self.assertEqual(set(metadata['tags']), set(new_metadata['tags']))

    def test_search_memories_with_query(self):
        """
        Tests memory search functionality with various test cases.
        
        API: search_memories
        Parameters:
            - agent_name: Name of the test agent
            - query: Search string ("meeting")
            - k: Number of results to return (3)
            - base_url: Server endpoint
        
        Verifies:
            - Success response
            - Result count <= k
            - Relevance of search results
        """
        # Create a few memories for testing search
        memory_contents = [
            "Meeting notes: Discussed Q1 goals and objectives",
            "Project roadmap for the next quarter",
            "Technical documentation for the memory system",
            "Notes from the team retrospective meeting"
        ]
        
        for i, content in enumerate(memory_contents):
            metadata = {
                "tags": ["test", f"tag{i}"],
                "priority": "medium" if i % 2 == 0 else "high"
            }
            
            response = create_memory(
                agent_name=self.agent_name,
                content=content,
                metadata=metadata,
                base_url=self.base_url
            )
            
            resp_data = self._get_response_data(response)
            if resp_data.get('success') and resp_data.get('memory_id'):
                self.search_memory_ids.append(resp_data.get('memory_id'))
        
        print(f"\nCreated {len(self.search_memory_ids)} test memories for search")
        
        # Search for memories
        search_query = "meeting"
        k = 3
        
        response = search_memories(
            agent_name=self.agent_name,
            query=search_query,
            k=k,
            base_url=self.base_url
        )
        
        print(f"\nSearch Memories Response (query='{search_query}', k={k}):")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Assertions
        self.assertTrue(resp_data.get('success'))
        search_results = resp_data.get('search_results', [])
        self.assertIsNotNone(search_results)
        
        # We should find at least one result with "meeting" in the content
        if search_results:
            self.assertLessEqual(len(search_results), k)
            found_meeting = False
            for result in search_results:
                if "meeting" in result.get('content', '').lower():
                    found_meeting = True
                    break
            self.assertTrue(found_meeting, "Expected to find at least one result containing 'meeting'")

    def test_delete_memory_by_id(self):
        """
        Tests memory deletion and verification of deletion.
        
        API: delete_memory, get_memory
        Parameters:
            - agent_name: Name of the test agent
            - memory_id: ID of memory to delete
            - base_url: Server endpoint
        
        Verifies:
            - Success response for deletion
            - Failed read attempt after deletion
        """
        # First create a memory if we don't have one
        if not self.memory_id:
            create_response = self.create_basic_memory()
            create_resp_data = self._get_response_data(create_response)
            self.memory_id = create_resp_data.get('memory_id')
            if not self.memory_id:
                self.fail("Failed to create test memory")
        
        memory_id_to_delete = self.memory_id
        
        response = delete_memory(
            agent_name=self.agent_name,
            memory_id=memory_id_to_delete,
            base_url=self.base_url
        )
        
        print("\nDelete Memory Response:")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Verify the deletion by trying to read the memory
        read_response = get_memory(
            agent_name=self.agent_name,
            memory_id=memory_id_to_delete,
            base_url=self.base_url
        )
        
        print("\nVerify Deletion (Read Memory):")
        print(f"  Response: {read_response}")
        
        read_resp_data = self._get_response_data(read_response)
        
        # Clear memory_id since we've deleted it
        self.memory_id = None
        
        # Assertions
        self.assertTrue(resp_data.get('success'))
        # Reading a deleted memory should fail
        self.assertFalse(read_resp_data.get('success'))
        self.assertIsNotNone(read_resp_data.get('error'))

    def test_handle_memory_errors(self):
        """
        Tests error handling for invalid memory operations.
        
        API: get_memory
        Parameters:
            - agent_name: Name of the test agent
            - memory_id: Non-existent ID
            - base_url: Server endpoint
        
        Verifies:
            - Appropriate error response
            - Error message presence
        """
        non_existent_memory = "non_existent_memory_id_12345"
        
        response = get_memory(
            agent_name=self.agent_name,
            memory_id=non_existent_memory,
            base_url=self.base_url
        )
        
        print("\nError Handling Test Response:")
        print(f"  Response: {response}")
        
        resp_data = self._get_response_data(response)
        
        # Assertions
        self.assertFalse(resp_data.get('success'))
        self.assertIsNotNone(resp_data.get('error'))


if __name__ == "__main__":
    unittest.main()
