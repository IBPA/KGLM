import pandas as pd

# file names
train_data_filename = './raw_data/train.txt'
val_data_filename = './raw_data/valid.txt'
test_data_filename = './raw_data/test.txt'

entity2text_filename = './raw_data/entity2textlong.txt'
relation2text_filename = './raw_data/relation2text.txt'

# load data
df_train_data = pd.read_csv(
    train_data_filename, sep='\t', names=['head', 'relation', 'tail'])
df_val_data = pd.read_csv(
    val_data_filename, sep='\t', names=['head', 'relation', 'tail'])
df_test_data = pd.read_csv(
    test_data_filename, sep='\t', names=['head', 'relation', 'tail'])

df_val_data['label'] = 1
df_test_data['label'] = 1

print('df_train_data.shape(): ', df_train_data.shape)
print('df_val_data.shape(): ', df_val_data.shape)
print('df_test_data.shape(): ', df_test_data.shape)

df_entities = pd.read_csv(
    entity2text_filename, sep='\t', names=['original_name', 'processed_name'])
df_entities['id'] = df_entities.index.to_series().apply(lambda x: 'e'+str(x))
df_entities['processed_name'] = df_entities['processed_name'].replace(
    {r'\\n': ' ', r'\\t': ' '}, regex=True)

# Get only the first sentence.
df_entities['processed_name'] = df_entities['processed_name'].apply(
    lambda x: x.split('. ')[0].rstrip())

df_relations = pd.read_csv(
    relation2text_filename, sep='\t', names=['original_name', 'processed_name'])
df_relations['id'] = df_relations.index.to_series().apply(lambda x: 'r'+str(x))

##########################
# entities and relations #
##########################
# save entities and relations
df_entities.to_csv('./processed_data/entities.txt', sep='\t', index=False)
df_relations.to_csv('./processed_data/relations.txt', sep='\t', index=False)

######################
# process data split #
######################
# replace original entity, relation names to processed names
entities_lookup = dict(zip(df_entities['original_name'], df_entities['id']))
relations_lookup = dict(zip(df_relations['original_name'], df_relations['id']))

df_train_data['head'] = df_train_data['head'].apply(lambda x: entities_lookup[x])
df_train_data['relation'] = df_train_data['relation'].apply(lambda x: relations_lookup[x])
df_train_data['tail'] = df_train_data['tail'].apply(lambda x: entities_lookup[x])
df_train_data['label'] = 1

df_val_data['head'] = df_val_data['head'].apply(lambda x: entities_lookup[x])
df_val_data['relation'] = df_val_data['relation'].apply(lambda x: relations_lookup[x])
df_val_data['tail'] = df_val_data['tail'].apply(lambda x: entities_lookup[x])

df_test_data['head'] = df_test_data['head'].apply(lambda x: entities_lookup[x])
df_test_data['relation'] = df_test_data['relation'].apply(lambda x: relations_lookup[x])
df_test_data['tail'] = df_test_data['tail'].apply(lambda x: entities_lookup[x])

# save
df_train_data.to_csv('./processed_data/train.txt', sep='\t', index=False)
df_val_data.to_csv('./processed_data/val.txt', sep='\t', index=False)
df_test_data.to_csv('./processed_data/test.txt', sep='\t', index=False)
