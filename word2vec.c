//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>

#define MAX_STRING 10000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30 * 1000 * 1000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  bool fixed;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_word_vectors_file[MAX_STRING], output_context_vectors_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char fixed_word_vectors_file[MAX_STRING], fixed_context_vectors_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, type = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 10000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int window_offset, window_layer_size;

int window_hidden_size = 500; 

int hs = 0, negative = 5, no_header = 0;
const int table_size = 1e8;
int *table;

//constrastive negative sampling
char negative_classes_file[MAX_STRING];
int *word_to_group;
int *group_to_table; //group_size*table_size
int class_number;

real hardTanh(real x){
	if(x>=1){
		return 1;
	}
	else if(x<=-1){
		return -1;
	}
	else{
		return x;
	}
}

real dHardTanh(real x, real g){
	if(x > 1 && g > 0){
		return 0;
	}
	if(x < -1 && g < 0){
		return 0;
	}
	return 1;
}

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab[vocab_size].fixed = false;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 10000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      vocab[b].fixed = vocab[a].fixed;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void ReadVectors(real *word_vectors, char *embeddings_filename) {
  FILE * fp = NULL;
  char *line = NULL;
  char *buffer = NULL;
  size_t len = 0;
  int vector_size = 0;
  ssize_t read;
  bool first_line = true;
  int j;

  if (embeddings_filename[0] == 0) { return; }

  fp = fopen(embeddings_filename, "r");
  if (fp == NULL) exit(EXIT_FAILURE);

  while ((read = getline(&line, &len, fp)) != -1) {
    char *first_field = NULL;
    long long i;
    // assert read != 0
    // discard the first line in the embeddings file (metadata)
    if (first_line) { 
      first_line = false; 
      // ignore the number of words
      buffer = strtok(line, " ");
      assert(buffer != NULL);
      // read the vector size
      buffer = strtok(NULL, " ");
      assert(buffer != NULL);
      vector_size = atoi(buffer);
      assert(vector_size > 0);
      continue;
    }
    
    // each line consists of the word and its embeddings. only read the word.
    first_field = strtok(line, " ");
    assert(first_field != NULL);
    i = SearchVocab(first_field);
    
    // ignore words which are not already in the vocabulary
    if (i == -1) { continue; }

    // now read the vector one value at a time.
    for (j = 0; j < vector_size; ++j) {
      buffer = strtok(NULL, " ");
      assert(buffer != NULL);
      real value = atof(buffer);
      word_vectors[i * layer1_size + j] = value;
    }
  }
  fclose(fp);
}

// add the words in fixed_word_vectors_file to the vocabulary.
void ReadFixedEmbeddingsVocab() {
  FILE * fp = NULL;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  printf("entering ReadFixedEmbeddingsVocab()\n");

  // do nothing if the file isn't specified.
  if (fixed_word_vectors_file[0] == 0) return;

  // assume it's a text mode embeddings file
  fp = fopen(fixed_word_vectors_file, "r");
  if (fp == NULL) exit(EXIT_FAILURE);

  bool first_line = true;
  while ((read = getline(&line, &len, fp)) != -1) {
    char * first_field = NULL;
    long long i;
    // discard the first line in the embeddings file (metadata)
    if (first_line) { first_line = false; continue; }
    // each line consists of the word and its embeddings. only read the word.
    first_field = strtok(line, " ");
    i = SearchVocab(first_field);
    // ignore new words and update the count of existing words so that they won't be pruned.
    if (i == -1) {
      // do nothing.
      continue;
    } else {
      vocab[i].cn = fmax(min_count, vocab[i].cn);
      vocab[i].fixed = true;
    }
  }
  fclose(fp);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
      vocab[a].fixed = false;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  ReadFixedEmbeddingsVocab();
  printf("will sort vocab...\n");
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  ReadFixedEmbeddingsVocab();
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitClassUnigramTable() {
  long long a,c;
  printf("loading class unigrams \n");
  FILE *fin = fopen(negative_classes_file, "rb");
  if (fin == NULL) {
    printf("ERROR: class file not found!\n");
    exit(1);
  }
  word_to_group = (int *)malloc(vocab_size * sizeof(int));
  for(a = 0; a < vocab_size; a++) word_to_group[a] = -1;
  char class[MAX_STRING];
  char prev_class[MAX_STRING];
  prev_class[0] = 0;
  char word[MAX_STRING];
  class_number = -1;
  while (1) {
    if (feof(fin)) break;
    ReadWord(class, fin);
    ReadWord(word, fin);
    int word_index = SearchVocab(word);
    if (word_index != -1){
      if(strcmp(class, prev_class) != 0){
        class_number++;
        strcpy(prev_class, class);
      }
      word_to_group[word_index] = class_number;
    }
    ReadWord(word, fin);
  }
  class_number++;
  fclose(fin);
  
  group_to_table = (int *)malloc(table_size * class_number * sizeof(int)); 
  long long train_words_pow = 0;
  real d1, power = 0.75;
  
  for(c = 0; c < class_number; c++){
    long long offset = c * table_size;
    train_words_pow = 0;
    for (a = 0; a < vocab_size; a++) if(word_to_group[a] == c) train_words_pow += pow(vocab[a].cn, power);
    int i = 0;
    while(word_to_group[i]!=c && i < vocab_size) i++;
    d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
      //printf("index %lld , word %d\n", a, i);
      group_to_table[offset + a] = i;
      if (a / (real)table_size > d1) {
        i++;
        while(word_to_group[i]!=c && i < vocab_size) i++;
        d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
      }
      if (i >= vocab_size) while(word_to_group[i]!=c && i >= 0) i--;
    }
  }
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  window_layer_size = layer1_size * window * 2;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    // Initialize syn1neg
    for (a = 0; a < vocab_size; a++) {
      for (b = 0; b < layer1_size; b++) {
        syn1neg[a * layer1_size + b] = 0;
      }
    }
    // Overwrite syn1neg for words in the fixed vocabulary
    ReadVectors(syn1neg, fixed_context_vectors_file);
  }
  // for each word in the vocab
  for (a = 0; a < vocab_size; a++) {
    // for each element in its vector representation
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      // initialize syn0
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
  }
  // Overwrite syn0 for words in the fixed vocabulary
  ReadVectors(syn0, fixed_word_vectors_file);
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  int input_len_1 = (type == 2 || type == 4)? window_layer_size : layer1_size;
  real *neu1e = (real *)calloc(input_len_1, sizeof(real));

  FILE *fi = fopen(train_file, "rb");

  // Each thread starts reading from a different part of the file.
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  // keep processing tokens from the training data until a check on the number of tokens processed
  // followed by another check on the number of passes over training data breaks the loop.
  while (1) {

    // every 10K updates, do some housekeeping such as updating the global count of processed tokens
    // (shard by all threads), print a progress report, update the learning rate (shared by all threads)
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // linearly decrease the learning rate (once after 1000 updates).
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    // sentence_length is reset before the first pass over the training data, after processing
    // all tokens in a sentence, and again after the thread processes the number of tokens it's
    //  supposed to process in each pass.
    if (sentence_length == 0) {
      // Read next token sequence from the train file.
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        // skip words which don't belong to the vocabulary. 
        if (word == -1) continue;
        word_count++;
        // word == 0 indicates a line break
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          // skip more frequent words more often
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        // ignore later words in sentences which are too long.
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    // (approx) finished processing my share of tokens in this pass over the training data
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      // if done with all my iterations, quit the loop.
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    // this is the "middle word", at sentence_position in the sentence.
    word = sen[sentence_position];
    
    if (word == -1) continue;

    // zero initialize the neu* (real) vectors.
    for (c = 0; c < input_len_1; c++) neu1e[c] = 0;    

    // let window = 5, then b = 0, 1, 2, 3, or 4.
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;

    if(type==1) {  //train skip-gram
      // loop over context words
      // let window = 5, b = 3, then a loops over 3,4,6,7,8
      // which correspond to context words at relative positions: -2, -1, 1, 2, 3
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          // the absolute position in the sentence for the current context word
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          // the index of the current context word
          last_word = sen[c];
          if (last_word == -1) continue;
          // the embedding of last_word starts at index l1 in the syn0 array
          l1 = last_word * layer1_size;
          // now, reuse c to loop over (and reset) the neu1e array
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          // NEGATIVE SAMPLING
          // loop over samples
          if (negative > 0) for (d = 0; d < negative + 1; d++) {

              // in the first iteration, target is the correct word in the middle, with label = 1
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                // in the remaining iterations, target is a random word from vocab, with label = 0
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if(word_to_group != NULL && word_to_group[word] != -1){
                  target = word;
                  while(target == word) {
                    target = group_to_table[word_to_group[word]*table_size + (next_random >> 16) % table_size];
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                  }
                  //printf("negative sampling %lld for word %s returned %s\n", d, vocab[word].word, vocab[target].word);
                }
                else{
                  target = table[(next_random >> 16) % table_size];
                }
                // fix bad samples
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              
              // the index of the embedding of target word in the syn0 array.
              l2 = target * layer1_size;
              
              // f = syn0[context] .* syn1neg[target]
              // f is the dot product between the context word's embedding (in syn0)
              // and the target word's embedding in syn1neg
              f = 0;
              for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
              
              // compute the scalar multiplier of the gradient update, which includes the sign
              if (f > MAX_EXP) 
                g = (label - 1) * alpha;
              else if (f < -MAX_EXP) 
                g = (label - 0) * alpha;
              else 
                g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          
              // for the current context word, neu1e accumulates the updates that should be later added to 
              // the (word embedding) of the context word in syn0
              for (c = 0; c < layer1_size; c++) {
                neu1e[c] += g * syn1neg[c + l2];
              }

              // Update the (context embedding) for the current target word in syn1neg
              if (!vocab[target].fixed || !fixed_context_vectors_file[0]) {
                for (c = 0; c < layer1_size; c++) { 
                  syn1neg[c + l2] += g * syn0[c + l1]; 
                }
              }
            }

          // Learn weights input -> hidden
          // update the (word embedding) for the current context word in syn0
          if (!vocab[last_word].fixed) {
            for (c = 0; c < layer1_size; c++) { 
              syn0[c + l1] += neu1e[c]; 
            }
          }
        }
    }
    else{
      printf("unknown type %i", type);
      exit(0);
    }
    sentence_position++;
    // done reading this sentence.
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) { ReadVocab(); }
  else { LearnVocabFromTrainFile(); }
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_word_vectors_file[0] == 0) {
    printf("output file not specified.\n");
    return;
  }
  printf("entering InitNet()...\n");
  InitNet();
  printf("exited InitNet()\n");
  if (negative > 0) InitUnigramTable();
  if (negative_classes_file[0] != 0) InitClassUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  // Save the word vectors
  fo = fopen(output_word_vectors_file, "wb");
  if (classes == 0) {
    if (!no_header) {
      fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    }
    for (a = 0; a < vocab_size; a++) {
      // skip words which we already have embeddings for
      if (vocab[a].fixed) { continue; }
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
  fclose(fo);

  // Save the context vectors
  fo = fopen(output_context_vectors_file, "wb");
  if (classes == 0) {
    if (!no_header) {
      fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    }
    for (a = 0; a < vocab_size; a++) {
      // skip contexts which we already have embeddings for
      if (vocab[a].fixed && fixed_context_vectors_file[0]) { continue; }
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t-negative-classes <file>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-type <int>\n");
    printf("\t\tType of embeddings (0 for cbow, 1 for skipngram, 2 for cwindow, 3 for structured skipngram, 4 for senna type)\n");
    printf("\t-fix-embeddings <file>\n");
    printf("\t\tUse the provided embeddings to induce embeddings for new words not specified in this file.\n");
    printf("\t-no-header <int>\n");
    printf("\t\tDon't write the header line in output embeddings file(s)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -type 1 -iter 3\n\n");
    return 0;
  }
  output_word_vectors_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  negative_classes_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-fix-embeddings", argc, argv)) > 0) {
    strcpy(fixed_word_vectors_file, argv[i + 1]);
    if (access( fixed_word_vectors_file, F_OK ) == -1) {
      // fixed word embeddings file does not exist
      fixed_word_vectors_file[0] = 0;
    }
    strcpy(fixed_context_vectors_file, fixed_word_vectors_file);
    strcat(fixed_context_vectors_file, ".context");
    if (access( fixed_context_vectors_file, F_OK ) == -1) {
      // fixed context embeddings file does not exist
      fixed_context_vectors_file[0] = 0;
    }
  }
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-type", argc, argv)) > 0) type = atoi(argv[i + 1]);
  if (type==0 || type==2 || type==4) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_word_vectors_file, argv[i + 1]);
    strcpy(output_context_vectors_file, output_word_vectors_file);
    strcat(output_context_vectors_file, ".context");
  }
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-no-header", argc, argv)) > 0) no_header = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative-classes", argc, argv)) > 0) strcpy(negative_classes_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
