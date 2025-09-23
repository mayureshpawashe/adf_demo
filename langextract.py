import os
import logging
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import textwrap

# LLMSherpa for intelligent PDF reading
from llmsherpa.readers import LayoutPDFReader

# Direct Google Gemini API
import google.generativeai as genai

# Neo4j for knowledge graph storage
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Triple:
    """Knowledge triple extracted from LLMSherpa + Gemini pipeline"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_chunk: Optional[str] = None
    timestamp: Optional[str] = None
    chunk_id: Optional[int] = None
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_type: Optional[str] = None
    extraction_text: Optional[str] = None
    extraction_class: Optional[str] = None

@dataclass
class ProcessingStats:
    """Comprehensive processing statistics"""
    total_pdf_pages: int = 0
    total_llmsherpa_chunks: int = 0
    processed_chunks: int = 0
    total_gemini_extractions: int = 0
    total_triples: int = 0
    stored_triples: int = 0
    failed_triples: int = 0
    unique_relationships: int = 0
    processing_time_seconds: float = 0.0

class Config:
    """Configuration for LLMSherpa + Gemini + Neo4j pipeline"""
    
    # Current date and time as specified
    CURRENT_UTC_TIME = "2025-08-05 07:40:00"
    CURRENT_USER_LOGIN = "mayureshpawashe"
    
    # LLMSherpa settings - using local Docker instance
    LLMSHERPA_API_URL = "http://localhost:5010/api/parseDocument?renderFormat=all"
    
    # Neo4j settings
    NEO4J_URI = os.getenv("NEO4J_URI_LLMSHERPA", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_LLMSHERPA", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_LLMSHERPA")

    # Google API Key for Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Processing settings
    GEMINI_MODEL = "gemini-2.5-pro"
    MAX_CHUNK_SIZE = 2500
    MAX_CHUNKS_TO_PROCESS = 20
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        missing = []
        
        if not cls.NEO4J_PASSWORD:
            missing.append("NEO4J_PASSWORD_LLMSHERPA")
            
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
            
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

class LLMSherpaGeminiDynamicProcessor:
    """LLMSherpa + Gemini processor with DYNAMIC relationship extraction"""
    
    def __init__(self):
        Config.validate()
        
        # Test LLMSherpa connection
        self._test_llmsherpa_connection()
        
        # Initialize LLMSherpa PDF reader
        self.pdf_reader = LayoutPDFReader(Config.LLMSHERPA_API_URL)
        
        # Configure Google Gemini API
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        
        # Configure safety settings to be more permissive for business content
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT", 
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        self.gemini_model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            safety_settings=self.safety_settings
        )
        
        # Initialize Neo4j
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        
        # Setup extraction prompt - similar to llmsherpaapitest approach
        self._setup_dynamic_extraction_prompt()
        
        # Triple pattern for extracting relationship patterns
        self.triple_pattern = re.compile(r"\((.*?)\)\s*-\[(.*?)\]->\s*\((.*?)\)")
        
        # Initialize tracking
        self.discovered_relationships = set()
        self.stats = ProcessingStats()
        
        self._test_connections()
        logger.info("✅ LLMSherpa + Gemini 2.5 Pro + Neo4j Pipeline initialized with DYNAMIC relationships")

    def _test_llmsherpa_connection(self):
        """Test local LLMSherpa Docker connection"""
        import requests
        
        try:
            test_url = "http://localhost:5010/api/parseDocument"
            response = requests.head(test_url, timeout=10)
            
            if response.status_code == 405 and 'POST' in response.headers.get('Allow', ''):
                logger.info("✅ LLMSherpa Docker connection successful")
            else:
                logger.warning(f"⚠️ Unexpected LLMSherpa response: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to LLMSherpa Docker")
            logger.error("💡 Start with: docker run -d -p 5010:5001 jamesmtc/nlm-ingestor:amd64")
            raise Exception("LLMSherpa Docker not accessible")
        except Exception as e:
            logger.error(f"❌ LLMSherpa connection test failed: {e}")
            raise

    def _setup_dynamic_extraction_prompt(self):
        """Setup dynamic extraction prompt for Gemini API calls"""
        
        # This prompt is similar to the one in llmsherpaapitest but adapted for Gemini
        self.extraction_prompt = textwrap.dedent(f"""\
            You are an expert knowledge extraction AI that creates DYNAMIC, DESCRIPTIVE relationships from text.

            Extract meaningful relationships as triples in this EXACT format:
            (Subject) -[Relation]-> (Object)

            IMPORTANT: Be CREATIVE and DESCRIPTIVE with relationship names. Use the MOST SPECIFIC and MEANINGFUL relationship that captures the exact nature of the connection.

            Examples of DYNAMIC relationships:
            - (Tesla) -[REVOLUTIONIZED]-> (Electric Vehicle Industry)
            - (COVID-19) -[ACCELERATED_ADOPTION_OF]-> (Remote Work)
            - (Netflix) -[DISRUPTED_STREAMING_MODEL_OF]-> (Traditional Television)
            - (Steve Jobs) -[CO_FOUNDED_AND_TRANSFORMED]-> (Apple Inc)
            - (Amazon) -[DOMINATES_CLOUD_INFRASTRUCTURE_THROUGH]-> (AWS)
            - (OpenAI) -[PIONEERED_BREAKTHROUGH_IN]-> (Large Language Models)
            - (Zoom) -[BECAME_ESSENTIAL_PLATFORM_FOR]-> (Virtual Meetings)
            - (SpaceX) -[ACHIEVED_FIRST_SUCCESSFUL]-> (Rocket Reusability)

            Guidelines:
            - Use DESCRIPTIVE relationship names that capture the exact nature of the connection
            - Include action words, outcomes, and context in the relationship name
            - Use underscores to connect multiple words: REVOLUTIONIZED_APPROACH_TO
            - Focus on unique, specific business dynamics and innovations
            - Capture temporal aspects: PIONEERED, ACCELERATED, TRANSFORMED, DISRUPTED
            - Include magnitude: DOMINATES, LEADS, PIONEERED, BREAKTHROUGH
            - Be creative but accurate to the source text
            - Maximum 30 triples per chunk
            - One triple per line

            Current Time: {Config.CURRENT_UTC_TIME}
            User: {Config.CURRENT_USER_LOGIN}

            Text content to analyze:
            """)

    def _test_connections(self):
        """Test all pipeline connections"""
        try:
            # Test Neo4j
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("✅ Neo4j connection successful")
                else:
                    raise Exception("Neo4j test query failed")
                    
            # Test Gemini API with safe content
            try:
                logger.info(f"🧪 Testing Gemini 2.5 Pro API with safe content...")
                
                test_prompt = self.extraction_prompt + "\n\nApple Inc. develops consumer electronics including the iPhone smartphone. The company operates retail stores worldwide and provides customer support services."
                
                response = self.gemini_model.generate_content(
                    test_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        top_p=0.95,
                        max_output_tokens=1024,
                    )
                )
                
                if response and hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    if candidate.finish_reason == 1 and hasattr(candidate, 'content') and candidate.content.parts:
                        logger.info("✅ Gemini 2.5 Pro API connection successful")
                        logger.info(f"📝 Test response: {len(candidate.content.parts[0].text)} characters")
                    else:
                        logger.warning(f"⚠️ Gemini test finished with reason: {candidate.finish_reason}")
                        logger.info("✅ Gemini API connected (will proceed with processing)")
                else:
                    logger.warning("⚠️ Unexpected Gemini response format")
                    logger.info("✅ Gemini API connected (will proceed with processing)")
                
            except Exception as e:
                logger.error(f"❌ Gemini API test failed: {e}")
                raise Exception(f"Gemini API connection failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Connection tests failed: {e}")
            raise Exception(f"Pipeline connection failure: {e}")

    def process_pdf_complete_pipeline(self, pdf_path: str) -> Dict[str, Any]:
        """Complete pipeline: LLMSherpa → Gemini → Neo4j with Dynamic Relationships"""
        start_time = datetime.now()
        
        logger.info(f"🚀 LLMSherpa + Gemini 2.5 Pro DYNAMIC Relationship Pipeline Started")
        logger.info(f"📅 UTC Time: {Config.CURRENT_UTC_TIME}")
        logger.info(f"👤 User: {Config.CURRENT_USER_LOGIN}")
        logger.info(f"📄 PDF: '{pdf_path}'")
        logger.info(f"🤖 Model: {Config.GEMINI_MODEL}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        try:
            # STEP 1: LLMSherpa intelligent PDF reading
            logger.info(f"📄 STEP 1: LLMSherpa intelligent PDF parsing")
            doc = self.pdf_reader.read_pdf(pdf_path)
            llmsherpa_chunks = self._extract_structured_chunks_with_llmsherpa(doc, pdf_name)
            self.stats.total_llmsherpa_chunks = len(llmsherpa_chunks)
            
            if not llmsherpa_chunks:
                logger.warning(f"⚠️ No chunks extracted by LLMSherpa from {pdf_name}")
                return self._build_empty_result(pdf_name)
            
            # STEP 2: Optimize chunks for Gemini processing
            logger.info(f"🔧 STEP 2: Optimizing chunks for Gemini processing")
            optimized_chunks = self._optimize_chunks_for_gemini(llmsherpa_chunks)
            self.stats.processed_chunks = len(optimized_chunks)
            
            # STEP 3: Dynamic relationship extraction with Gemini
            logger.info(f"🎯 STEP 3: Dynamic relationship extraction with Gemini 2.5 Pro")
            extracted_triples = self._extract_dynamic_relationships_with_gemini(optimized_chunks)
            self.stats.total_triples = len(extracted_triples)
            
            # STEP 4: Store in Neo4j with DYNAMIC RELATIONSHIPS
            logger.info(f"🕸️ STEP 4: Storing DYNAMIC relationships in Neo4j knowledge graph")
            neo4j_stats = self._store_dynamic_relationships_in_neo4j(extracted_triples)
            self.stats.stored_triples = neo4j_stats['inserted']
            self.stats.failed_triples = neo4j_stats['failed']
            
            # STEP 5: Generate comprehensive results
            end_time = datetime.now()
            self.stats.processing_time_seconds = (end_time - start_time).total_seconds()
            self.stats.unique_relationships = len(self.discovered_relationships)
            
            result = self._build_comprehensive_pipeline_result(
                pdf_name, pdf_path, llmsherpa_chunks, optimized_chunks, 
                extracted_triples, neo4j_stats
            )
            
            logger.info(f"✅ Complete dynamic pipeline finished successfully!")
            logger.info(f"📊 Results: {self.stats.total_triples} triples, {self.stats.unique_relationships} relationship types")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            raise

    def _extract_structured_chunks_with_llmsherpa(self, doc, pdf_name: str) -> List[Dict]:
        """Extract structured chunks using LLMSherpa's intelligent parsing"""
        chunks = []
        
        try:
            # Get LLMSherpa's intelligent document structure
            doc_chunks = doc.chunks()
            logger.info(f"📄 LLMSherpa extracted {len(doc_chunks)} intelligent chunks")
            
            # Track pages
            try:
                total_pages = len(doc.pages()) if hasattr(doc, 'pages') else 0
                self.stats.total_pdf_pages = total_pages
                logger.info(f"📖 PDF has {total_pages} pages")
            except:
                self.stats.total_pdf_pages = 0
            
            for i, chunk in enumerate(doc_chunks):
                try:
                    content = chunk.to_context_text()
                    
                    # Skip low-quality content
                    if len(content.strip()) < 100 or self._is_low_quality_content(content):
                        continue
                    
                    # Extract enhanced metadata from LLMSherpa
                    section = self._extract_section_from_chunk(chunk, i)
                    page_idx = getattr(chunk, 'page_idx', 0)
                    chunk_type = getattr(chunk, 'tag', 'paragraph')
                    
                    # Calculate content quality for prioritization
                    quality_score = self._calculate_content_quality_score(content)
                    
                    chunk_data = {
                        'content': content.strip(),
                        'section': section,
                        'page': page_idx,
                        'chunk_type': chunk_type,
                        'chunk_id': i,
                        'source_pdf': pdf_name,
                        'quality_score': quality_score,
                        'char_count': len(content)
                    }
                    
                    chunks.append(chunk_data)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing LLMSherpa chunk {i}: {e}")
                    continue
            
            # Process tables separately with LLMSherpa
            self._extract_tables_with_llmsherpa(doc, chunks, pdf_name)
            
        except Exception as e:
            logger.error(f"❌ Error in LLMSherpa chunk extraction: {e}")
        
        # Sort by quality score for better processing order
        chunks.sort(key=lambda x: x['quality_score'], reverse=True)
        
        logger.info(f"📝 LLMSherpa extracted {len(chunks)} high-quality chunks")
        return chunks

    def _optimize_chunks_for_gemini(self, llmsherpa_chunks: List[Dict]) -> List[Dict]:
        """Optimize LLMSherpa chunks for Gemini processing"""
        optimized_chunks = []
        
        for chunk in llmsherpa_chunks:
            content = chunk['content']
            
            # If chunk is too large, split intelligently
            if len(content) > Config.MAX_CHUNK_SIZE:
                sub_chunks = self._split_large_chunk_intelligently(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        # Prioritize high-quality chunks and limit for cost control
        optimized_chunks.sort(key=lambda x: x['quality_score'], reverse=True)
        final_chunks = optimized_chunks[:Config.MAX_CHUNKS_TO_PROCESS]
        
        logger.info(f"🔧 Optimized {len(llmsherpa_chunks)} LLMSherpa chunks to {len(final_chunks)} for Gemini processing")
        return final_chunks

    def _extract_dynamic_relationships_with_gemini(self, chunks: List[Dict]) -> List[Triple]:
        """Extract dynamic relationships using Gemini API calls"""
        extracted_triples = []
        total_extractions = 0
        successful_chunks = 0
        blocked_chunks = 0
        
        for i, chunk_data in enumerate(chunks):
            try:
                logger.info(f"🎯 Gemini processing chunk {i+1}/{len(chunks)}")
                logger.info(f"   Quality: {chunk_data.get('quality_score', 0):.2f}, "
                          f"Type: {chunk_data.get('chunk_type', 'unknown')}, "
                          f"Section: {chunk_data.get('section', 'unknown')[:40]}...")
                
                # Prepare dynamic prompt for Gemini
                full_prompt = self.extraction_prompt + "\n\n" + chunk_data['content']
                
                # Call Gemini API with enhanced error handling
                try:
                    response = self.gemini_model.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,  # Slightly higher for creativity
                            top_p=0.95,
                            max_output_tokens=1024,
                        )
                    )
                    
                    # Enhanced response handling
                    if response and hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        
                        if candidate.finish_reason == 1:  # STOP - successful completion
                            if hasattr(candidate, 'content') and candidate.content.parts:
                                response_text = candidate.content.parts[0].text
                                
                                # Parse Gemini response into triples
                                chunk_triples, chunk_extractions = self._parse_dynamic_triples(
                                    response_text.strip().splitlines(),
                                    chunk_data['content'][:200],
                                    chunk_data['chunk_id'],
                                    chunk_data.get('section', 'Unknown'),
                                    chunk_data
                                )
                                
                                extracted_triples.extend(chunk_triples)
                                total_extractions += chunk_extractions
                                successful_chunks += 1
                                
                                # Track discoveries
                                for triple in chunk_triples:
                                    self.discovered_relationships.add(triple.predicate)
                                
                                logger.info(f"✅ Gemini chunk {i+1}: {len(chunk_triples)} DYNAMIC relationships extracted")
                                
                                # Show sample relationships
                                if chunk_triples:
                                    for j, triple in enumerate(chunk_triples[:2]):
                                        logger.info(f"   🔗 {j+1}. ({triple.subject}) -[{triple.predicate}]-> ({triple.object})")
                            else:
                                logger.warning(f"⚠️ Empty response from Gemini for chunk {i+1}")
                        elif candidate.finish_reason == 2:  # SAFETY
                            logger.warning(f"⚠️ Gemini blocked chunk {i+1} due to safety filters")
                            blocked_chunks += 1
                        elif candidate.finish_reason == 3:  # RECITATION
                            logger.warning(f"⚠️ Gemini blocked chunk {i+1} due to recitation")
                            blocked_chunks += 1
                        else:
                            logger.warning(f"⚠️ Gemini chunk {i+1} finished with reason: {candidate.finish_reason}")
                    else:
                        logger.warning(f"⚠️ Unexpected Gemini response format for chunk {i+1}")
                
                except Exception as api_error:
                    logger.error(f"❌ Gemini API error on chunk {i+1}: {api_error}")
                    continue
                
            except Exception as e:
                logger.error(f"❌ General error on chunk {i+1}: {e}")
                continue
        
        self.stats.total_gemini_extractions = total_extractions
        logger.info(f"🎯 Gemini completed: {len(extracted_triples)} relationships from {successful_chunks} successful chunks")
        logger.info(f"⚠️ {blocked_chunks} chunks blocked by safety filters")
        return extracted_triples

    def _parse_dynamic_triples(self, raw_triples: List[str], source_chunk: str, chunk_id: int, section: str, chunk_data: Dict) -> tuple:
        """Parse and validate dynamic triple strings"""
        parsed_triples = []
        total_extractions = len(raw_triples)

        for raw_triple in raw_triples:
            try:
                triple = raw_triple.strip()
                
                # Skip empty lines or very short content
                if not triple or len(triple) < 15:
                    continue
                
                # Remove numbering if present
                if re.match(r'^\d+\.?\s*', triple):
                    triple = re.sub(r'^\d+\.?\s*', '', triple)
                
                match = self.triple_pattern.match(triple)
                if not match:
                    continue
                
                subj, rel, obj = match.groups()
                
                # Clean entities
                subj = subj.strip()
                rel = rel.strip().upper().replace(' ', '_').replace('-', '_')
                obj = obj.strip()
                
                # Basic validation
                if len(subj) > 1 and len(rel) > 1 and len(obj) > 1:
                    # Clean up relationship name
                    rel = re.sub(r'[^\w_]', '', rel)  # Remove special chars except underscore
                    rel = re.sub(r'_+', '_', rel)     # Replace multiple underscores with single
                    rel = rel.strip('_')              # Remove leading/trailing underscores
                    
                    if len(rel) > 0:
                        parsed_triple = Triple(
                            subject=subj,
                            predicate=rel,
                            object=obj,
                            confidence=0.9,
                            source_chunk=source_chunk,
                            timestamp=datetime.now().isoformat(),
                            chunk_id=chunk_id,
                            section=section,
                            page=chunk_data.get('page', 0),
                            chunk_type=chunk_data.get('chunk_type', 'paragraph'),
                            extraction_text=triple,
                            extraction_class='dynamic_relationship'
                        )
                        parsed_triples.append(parsed_triple)
                
            except Exception as e:
                logger.warning(f"⚠️ Error parsing dynamic triple: {e}")
                continue

        return parsed_triples, total_extractions

    def _sanitize_relationship_type(self, rel_type: str) -> str:
        """Sanitize relationship type for use in Neo4j"""
        # Remove invalid characters for Neo4j relationship types
        sanitized = re.sub(r'[^\w_]', '_', rel_type)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"REL_{sanitized}"
        
        # Limit length (Neo4j has practical limits)
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "DYNAMIC_RELATIONSHIP"
        
        return sanitized

    def _store_dynamic_relationships_in_neo4j(self, triples: List[Triple]) -> Dict[str, int]:
        """Store triples as DYNAMIC RELATIONSHIPS in Neo4j"""
        
        def create_dynamic_relationship_types(tx, batch_triples):
            """Create relationships with actual dynamic types using dynamic Cypher"""
            results = []
            
            for triple in batch_triples:
                # Create dynamic relationship type query
                sanitized_rel_type = self._sanitize_relationship_type(triple.predicate)
                
                query = f"""
                MERGE (a:Entity {{name: $subject}})
                MERGE (b:Entity {{name: $object}})
                CREATE (a)-[r:{sanitized_rel_type} {{
                    original_relationship_type: $predicate,
                    source_chunk: $source_chunk,
                    timestamp: $timestamp,
                    confidence: $confidence,
                    chunk_id: $chunk_id,
                    section: $section,
                    page: $page,
                    chunk_type: $chunk_type,
                    extraction_method: "llmsherpa_gemini_dynamic",
                    extraction_class: $extraction_class,
                    extraction_text: $extraction_text,
                    gemini_model: "{Config.GEMINI_MODEL}",
                    user_login: "{Config.CURRENT_USER_LOGIN}",
                    processing_utc: "{Config.CURRENT_UTC_TIME}"
                }}]->(b)
                
                SET a.last_updated = timestamp(),
                    a.extraction_method = "llmsherpa_gemini_dynamic",
                    a.user_login = "{Config.CURRENT_USER_LOGIN}",
                    b.last_updated = timestamp(),
                    b.extraction_method = "llmsherpa_gemini_dynamic",
                    b.user_login = "{Config.CURRENT_USER_LOGIN}"
                
                RETURN 1 as created
                """
                
                try:
                    result = tx.run(query, 
                        subject=triple.subject,
                        object=triple.object,
                        predicate=triple.predicate,
                        source_chunk=triple.source_chunk,
                        timestamp=triple.timestamp,
                        confidence=triple.confidence,
                        chunk_id=triple.chunk_id,
                        section=triple.section,
                        page=triple.page,
                        chunk_type=triple.chunk_type,
                        extraction_class=triple.extraction_class,
                        extraction_text=triple.extraction_text
                    )
                    
                    if result.single():
                        results.append(1)
                    else:
                        results.append(0)
                        
                except Exception as e:
                    logger.warning(f"Failed to create dynamic relationship {triple.predicate}: {e}")
                    results.append(0)
            
            return sum(results)

        def create_fallback_relationships(tx, batch_triples):
            """Fallback: Store as RELATED_TO with relationship_type property"""
            query = """
            UNWIND $triples AS triple
            MERGE (a:Entity {name: triple.subject})
            MERGE (b:Entity {name: triple.object})
            CREATE (a)-[r:RELATED_TO {
                relationship_type: triple.predicate,
                source_chunk: triple.source_chunk,
                timestamp: triple.timestamp,
                confidence: triple.confidence,
                chunk_id: triple.chunk_id,
                section: triple.section,
                page: triple.page,
                chunk_type: triple.chunk_type,
                extraction_method: "llmsherpa_gemini_dynamic_fallback",
                extraction_class: triple.extraction_class,
                extraction_text: triple.extraction_text,
                gemini_model: "gemini-2.5-pro",
                user_login: "mayureshpawashe",
                processing_utc: "2025-08-05 07:40:00"
            }]->(b)
            RETURN count(r) as created
            """
            
            result = tx.run(query, triples=[{
                'subject': t.subject,
                'predicate': t.predicate,
                'object': t.object,
                'source_chunk': t.source_chunk,
                'timestamp': t.timestamp,
                'confidence': t.confidence,
                'chunk_id': t.chunk_id,
                'section': t.section,
                'page': t.page,
                'chunk_type': t.chunk_type,
                'extraction_class': t.extraction_class,
                'extraction_text': t.extraction_text
            } for t in batch_triples])
            
            return result.single()['created']

        stats = {'inserted': 0, 'failed': 0}
        batch_size = 3  # Small batches for dynamic relationship creation
        
        logger.info(f"📤 Storing {len(triples)} DYNAMIC relationships in Neo4j")
        
        try:
            with self.driver.session() as session:
                for i in range(0, len(triples), batch_size):
                    batch = triples[i:i + batch_size]
                    
                    try:
                        # Try creating dynamic relationship types first
                        created = session.write_transaction(create_dynamic_relationship_types, batch)
                        stats['inserted'] += created
                        
                        logger.info(f"✅ DYNAMIC batch {i//batch_size + 1}: {created} relationships stored")
                        
                        for t in batch[:2]:
                            logger.info(f"   🔗 {t.subject} -[{t.predicate}]-> {t.object}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Dynamic types failed for batch {i//batch_size + 1}, using fallback: {e}")
                        
                        try:
                            # Fallback to RELATED_TO with properties
                            created = session.write_transaction(create_fallback_relationships, batch)
                            stats['inserted'] += created
                            logger.info(f"✅ Fallback: Batch {i//batch_size + 1}: Created {created} RELATED_TO relationships")
                        except Exception as fallback_error:
                            logger.error(f"❌ Both dynamic and fallback failed for batch {i//batch_size + 1}: {fallback_error}")
                            stats['failed'] += len(batch)
                            continue
                        
        except Exception as e:
            logger.error(f"❌ Neo4j connection error: {e}")
            raise
            
        logger.info(f"📊 DYNAMIC relationship storage complete: {stats['inserted']} stored, {stats['failed']} failed")
        return stats

    def _build_comprehensive_pipeline_result(self, pdf_name: str, pdf_path: str, 
                                           llmsherpa_chunks: List[Dict], optimized_chunks: List[Dict],
                                           triples: List[Triple], neo4j_stats: Dict) -> Dict[str, Any]:
        """Build comprehensive pipeline result"""
        
        extraction_class_distribution = {}
        relationship_distribution = {}
        
        for triple in triples:
            class_name = triple.extraction_class or 'unknown'
            extraction_class_distribution[class_name] = extraction_class_distribution.get(class_name, 0) + 1
            
            predicate = triple.predicate
            relationship_distribution[predicate] = relationship_distribution.get(predicate, 0) + 1
        
        result = {
            'pipeline_info': {
                'pipeline_name': 'LLMSherpa + Gemini 2.5 Pro + Neo4j DYNAMIC RELATIONSHIPS',
                'pdf_name': pdf_name,
                'pdf_path': pdf_path,
                'processing_utc': Config.CURRENT_UTC_TIME,
                'user_login': Config.CURRENT_USER_LOGIN,
                'gemini_model': Config.GEMINI_MODEL,
                'processing_timestamp': datetime.now().isoformat()
            },
            'processing_stats': {
                'total_pdf_pages': self.stats.total_pdf_pages,
                'llmsherpa_chunks_extracted': self.stats.total_llmsherpa_chunks,
                'chunks_optimized_for_gemini': self.stats.processed_chunks,
                'total_gemini_extractions': self.stats.total_gemini_extractions,
                'total_triples_generated': len(triples),
                'dynamic_relationships_stored_in_neo4j': neo4j_stats['inserted'],
                'failed_storage_attempts': neo4j_stats['failed'],
                'unique_relationship_types': len(self.discovered_relationships),
                'processing_time_seconds': self.stats.processing_time_seconds
            },
            'content_analysis': {
                'discovered_relationship_types': sorted(list(self.discovered_relationships)),
                'extraction_class_distribution': dict(sorted(extraction_class_distribution.items(), key=lambda x: x[1], reverse=True)),
                'dynamic_relationship_distribution': dict(sorted(relationship_distribution.items(), key=lambda x: x[1], reverse=True)[:20])
            },
            'sample_results': {
                'top_confidence_dynamic_relationships': [
                    {
                        'subject': t.subject,
                        'relationship_type': t.predicate,
                        'object': t.object,
                        'confidence': t.confidence,
                        'extraction_class': t.extraction_class,
                        'section': t.section
                    } for t in sorted(triples, key=lambda x: x.confidence, reverse=True)[:10]
                ]
            }
        }
        
        return result

    def _build_empty_result(self, pdf_name: str) -> Dict[str, Any]:
        return {
            'pipeline_info': {
                'pipeline_name': 'LLMSherpa + Gemini 2.5 Pro + Neo4j DYNAMIC RELATIONSHIPS',
                'pdf_name': pdf_name,
                'processing_utc': Config.CURRENT_UTC_TIME,
                'user_login': Config.CURRENT_USER_LOGIN,
                'gemini_model': Config.GEMINI_MODEL,
                'processing_timestamp': datetime.now().isoformat()
            },
            'processing_stats': {
                'total_pdf_pages': 0,
                'llmsherpa_chunks_extracted': 0,
                'chunks_optimized_for_gemini': 0,
                'total_gemini_extractions': 0,
                'total_triples_generated': 0,
                'dynamic_relationships_stored_in_neo4j': 0,
                'processing_time_seconds': 0
            },
            'error': 'No meaningful content extracted from PDF'
        }

    # Helper methods
    def _is_low_quality_content(self, content: str) -> bool:
        if len(content.strip()) < 100:
            return True
        words = content.split()
        if len(words) < 10:
            return True
        if len(set(words)) / len(words) < 0.4:
            return True
        alpha_ratio = sum(c.isalpha() for c in content) / len(content)
        if alpha_ratio < 0.5:
            return True
        return False

    def _calculate_content_quality_score(self, content: str) -> float:
        score = 0.0
        length = len(content)
        if 1000 <= length <= 2500:
            score += 0.3
        elif 500 <= length <= 1000 or 2500 <= length <= 3500:
            score += 0.2
        elif length > 200:
            score += 0.1
        
        words = content.split()
        if words and len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.3
        
        alpha_ratio = sum(c.isalpha() for c in content) / len(content) if content else 0
        score += alpha_ratio * 0.25
        
        sentences = content.count('.') + content.count('!') + content.count('?')
        if sentences > 0:
            avg_sentence_length = len(words) / sentences if words else 0
            if 10 <= avg_sentence_length <= 35:
                score += 0.15
        
        return min(score, 1.0)

    def _extract_section_from_chunk(self, chunk, index: int) -> str:
        section = getattr(chunk, 'parent_header', None)
        if not section or section.strip() == '':
            content = chunk.to_context_text()[:100].strip()
            if content.startswith(('CHAPTER', 'SECTION', 'PART', 'Chapter', 'Section')):
                section = content.split('\n')[0][:50]
            else:
                section = f"Section_{index+1}"
        return section

    def _extract_tables_with_llmsherpa(self, doc, chunks: List[Dict], pdf_name: str):
        try:
            tables = doc.tables()
            if tables:
                logger.info(f"📊 LLMSherpa processing {len(tables)} tables")
                for i, table in enumerate(tables):
                    try:
                        table_content = table.to_text()
                        if len(table_content.strip()) > 100:
                            quality_score = self._calculate_content_quality_score(table_content)
                            table_chunk = {
                                'content': table_content.strip(),
                                'section': f"Table_{i+1}",
                                'page': 0,
                                'chunk_type': 'table',
                                'chunk_id': len(chunks),
                                'source_pdf': pdf_name,
                                'quality_score': quality_score,
                                'char_count': len(table_content)
                            }
                            chunks.append(table_chunk)
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing table {i}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"⚠️ Error in LLMSherpa table extraction: {e}")

    def _split_large_chunk_intelligently(self, chunk: Dict) -> List[Dict]:
        sub_chunks = []
        content = chunk['content']
        max_size = Config.MAX_CHUNK_SIZE
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        current_chunk = ""
        chunk_counter = 0
        
        for para in paragraphs:
            if len(current_chunk + para) <= max_size:
                current_chunk += para + "\n\n" if current_chunk else para
            else:
                if current_chunk:
                    sub_chunk = chunk.copy()
                    sub_chunk['content'] = current_chunk.strip()
                    sub_chunk['chunk_id'] = f"{chunk['chunk_id']}_sub_{chunk_counter}"
                    sub_chunk['char_count'] = len(current_chunk)
                    sub_chunks.append(sub_chunk)
                    chunk_counter += 1
                current_chunk = para
        
        if current_chunk.strip():
            sub_chunk = chunk.copy()
            sub_chunk['content'] = current_chunk.strip()
            sub_chunk['chunk_id'] = f"{chunk['chunk_id']}_sub_{chunk_counter}"
            sub_chunk['char_count'] = len(current_chunk)
            sub_chunks.append(sub_chunk)
        
        return sub_chunks

    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("🔒 LLMSherpa + Gemini + Neo4j DYNAMIC Pipeline resources closed")

def main():
    """Main function for LLMSherpa + Gemini + Neo4j pipeline with Dynamic Relationships"""
    try:
        print("🚀 LLMSherpa + Gemini 2.5 Pro + Neo4j DYNAMIC RELATIONSHIPS Pipeline")
        print("="*100)
        print("📄 PDF Reading: LLMSherpa (Intelligent Structure)")
        print("🧠 Relationship Extraction: Gemini 2.5 Pro API with DYNAMIC Relationship Names")
        print("🕸️ Knowledge Graph: Neo4j with DYNAMIC RELATIONSHIP TYPES")
        print("="*100)
        print(f"👤 User: {Config.CURRENT_USER_LOGIN}")
        print(f"📅 UTC Time: {Config.CURRENT_UTC_TIME}")
        print(f"🤖 AI Model: {Config.GEMINI_MODEL}")
        print("🔗 Relationship Storage: DYNAMIC, DESCRIPTIVE RELATIONSHIP TYPES")
        print("="*100)
        
        # Get PDF path
        pdf_path = input("📁 Enter PDF file path: ").strip()
        
        if not pdf_path:
            print("⚠️ Please enter a valid PDF path.")
            return
        
        if not os.path.exists(pdf_path):
            print("❌ PDF file not found.")
            return
        
        # Initialize pipeline
        pipeline = LLMSherpaGeminiDynamicProcessor()
        
        try:
            print(f"\n🔄 Starting LLMSherpa + Gemini pipeline with DYNAMIC RELATIONSHIPS...")
            result = pipeline.process_pdf_complete_pipeline(pdf_path)
            
            # Display results
            print(f"\n📊 DYNAMIC RELATIONSHIPS PIPELINE RESULTS:")
            print(f"="*100)
            print(f"📄 PDF: {result['pipeline_info']['pdf_name']}")
            print(f"🤖 Model: {result['pipeline_info']['gemini_model']}")
            print(f"📖 Pages processed: {result['processing_stats']['total_pdf_pages']}")
            print(f"📄 LLMSherpa chunks: {result['processing_stats']['llmsherpa_chunks_extracted']}")
            print(f"🔧 Optimized chunks: {result['processing_stats']['chunks_optimized_for_gemini']}")
            print(f"🔍 Gemini extractions: {result['processing_stats']['total_gemini_extractions']}")
            print(f"🎯 Total triples: {result['processing_stats']['total_triples_generated']}")
            print(f"🕸️ DYNAMIC relationships stored: {result['processing_stats']['dynamic_relationships_stored_in_neo4j']}")
            print(f"⏱️ Processing time: {result['processing_stats']['processing_time_seconds']:.1f} seconds")
            print(f"🔗 Unique relationship types: {result['processing_stats']['unique_relationship_types']}")
            
            # Show DYNAMIC relationship types discovered
            if result['content_analysis']['discovered_relationship_types']:
                print(f"\n🔗 DYNAMIC RELATIONSHIP TYPES DISCOVERED:")
                for i, rel_type in enumerate(result['content_analysis']['discovered_relationship_types'][:15], 1):
                    count = result['content_analysis']['dynamic_relationship_distribution'].get(rel_type, 0)
                    print(f"   {i}. {rel_type}: {count} occurrences")
            
            # Show DYNAMIC relationships
            if result['sample_results']['top_confidence_dynamic_relationships']:
                print(f"\n💎 TOP DYNAMIC RELATIONSHIPS:")
                for i, rel in enumerate(result['sample_results']['top_confidence_dynamic_relationships'][:5], 1):
                    print(f"   {i}. ({rel['subject']}) -[{rel['relationship_type']}]-> ({rel['object']})")
                    print(f"      Confidence: {rel['confidence']:.2f}, Source: {rel['section'][:50]}")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"llmsherpa_gemini_dynamic_relationships_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {filename}")
            
            print(f"\n🎉 DYNAMIC RELATIONSHIPS Pipeline Processing Successful!")
            print(f"🔗 Created {result['processing_stats']['unique_relationship_types']} different relationship types")
            print(f"✨ Neo4j contains DESCRIPTIVE relationship types reflecting actual relationship meanings")
            
        finally:
            pipeline.close()
            
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        print(f"❌ Pipeline error occurred: {e}")

if __name__ == "__main__":
    main()