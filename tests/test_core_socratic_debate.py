--- a/tests/test_core_socratic_debate.py
+++ b/tests/test_core_socratic_debate.py
@@ -100,7 +100,8 @@
             model_name="gemini-2.5-flash-lite",  # Use a light model for tests
             domain="General",  # Use 'General' for simple questions
             persona_manager=persona_manager_instance,  # Pass the persona manager
             context_analyzer=mock_context_analyzer,
             token_tracker=mock_token_tracker,
             settings=mock_settings, # MODIFIED: Pass mock_settings
+            structured_codebase_context={}, # NEW: Add structured_codebase_context
+            raw_file_contents={"file1.py": "content"}, # NEW: Add raw_file_contents
         )
         # Ensure the conflict manager mock is assigned to the instance
         debate.conflict_manager = mock_conflict_manager
@@ -242,7 +243,8 @@
                 persona_manager=pm_for_test,
                 llm_provider=mock_gemini_provider_instance, # Pass the mocked provider
                 context_analyzer=mock_context_analyzer,
                 token_tracker=mock_token_tracker,
                 settings=mock_settings,
+                structured_codebase_context={}, # NEW: Add structured_codebase_context
+                raw_file_contents={"file1.py": "content"}, # NEW: Add raw_file_contents
             )
             debate_manager_instance.output_parser = mock_output_parser # Assign the mock parser
             return debate_manager_instance
@@ -300,7 +302,7 @@
     assert "Self_Improvement_Analyst_Output" in intermediate_steps
 
 def test_socratic_debate_malformed_output_triggers_conflict_manager(socratic_debate_instance, mock_gemini_provider, mock_conflict_manager):
-    """Tests that malformed output triggers the conflict manager."""
+    """Tests that malformed output triggers the conflict manager and that resolution is handled."""
     socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
         "Visionary_Generator", "Constructive_Critic", "Impartial_Arbitrator"
     ]
@@ -337,7 +339,7 @@
     # Assert that the resolved output from the conflict manager was used in the history
     # The history should contain an entry for Conflict_Resolution_Manager
     conflict_manager_turn = next((t for t in intermediate_steps.get('Debate_History', []) if t.get('persona') == 'Conflict_Resolution_Manager'), None)
-    assert conflict_manager_turn is not None, "Conflict resolution turn not found in history"
+    assert conflict_manager_turn is not None, "Conflict resolution manager turn not found in history"
     assert "Mock resolved output from conflict" in conflict_manager_turn['output'].get('resolved_output', {}).get('general_output', '')
 
 def test_socratic_debate_token_budget_exceeded(socratic_debate_instance, mock_gemini_provider, mock_token_tracker):
@@ -375,7 +377,7 @@
     assert mock_gemini_provider.generate.call_count == 2 # 3 personas in sequence
     # Assert that parse_and_validate was called for each turn
     assert socratic_debate_instance.output_parser.parse_and_validate.call_count == 3 # Parser called for each turn
 
 def test_execute_llm_turn_schema_validation_retry(socratic_debate_instance, mock_gemini_provider):
     """Tests that _execute_llm_turn retries on SchemaValidationError."""
     persona_name = "Constructive_Critic"
@@ -409,7 +411,7 @@
     assert socratic_debate_instance.output_parser.parse_and_validate.call_count == 2
     # Assert that the final output is the valid one
     assert output["CRITIQUE_SUMMARY"] == "Valid critique"
     # Assert that a malformed block for retry was recorded
     assert any(block['type'] == 'RETRYABLE_VALIDATION_ERROR' for block in socratic_debate_instance.intermediate_steps.get('malformed_blocks', []))
 
 def test_socratic_debate_self_analysis_flow(socratic_debate_instance, mock_gemini_provider, mock_persona_manager, mock_metrics_collector):
@@ -420,7 +422,7 @@
     mock_persona_manager.prompt_analyzer.is_self_analysis_prompt.return_value = True
     mock_persona_manager.persona_router.determine_persona_sequence.return_value = ["Self_Improvement_Analyst"]
 
     # Mock LLM response for Self_Improvement_Analyst
     mock_gemini_provider.generate.return_value = (
         """{"ANALYSIS_SUMMARY": "Self-analysis complete.", "IMPACTFUL_SUGGESTIONS": []}""",
@@ -430,7 +432,7 @@
     socratic_debate_instance.output_parser.parse_and_validate.side_effect = [
         # Context_Aware_Assistant (if in sequence)
         ({'general_overview': 'Context overview', 'malformed_blocks': []}),
         # Self_Improvement_Analyst
         ({'ANALYSIS_SUMMARY': 'Self-analysis complete.', 'IMPACTFUL_SUGGESTIONS': [], 'malformed_blocks': []}),
     ]
 
     final_answer, intermediate_steps = socratic_debate_instance.run_debate()
 
@@ -440,7 +442,7 @@
     assert "Self_Improvement_Analyst_Output" in intermediate_steps
 
 def test_socratic_debate_context_aware_assistant_turn(socratic_debate_instance, mock_gemini_provider, mock_context_analyzer):
-    """Tests the Context_Aware_Assistant turn when present in the sequence."""
+    """Tests the Context_Aware_Assistant turn when present in the sequence, ensuring context is passed."""
     socratic_debate_instance.persona_router.determine_persona_sequence.return_value = [
         "Context_Aware_Assistant", "Impartial_Arbitrator"
     ]
@@ -457,7 +459,7 @@
     ]
 
     final_answer, intermediate_steps = socratic_debate_instance.run_debate()
 
     assert "Context_Aware_Assistant_Output" in intermediate_steps
     assert intermediate_steps["Context_Aware_Assistant_Output"].get("general_overview") == "Context analysis output"
     mock_context_analyzer.find_relevant_files.assert_called_once()
     mock_context_analyzer.generate_context_summary.assert_called_once()